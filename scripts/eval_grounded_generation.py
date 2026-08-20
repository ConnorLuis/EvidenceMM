from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import yaml

from evidencemm.data_binding import sha256_file
from evidencemm.grounded_generation import generate_grounded, load_generator
from evidencemm.grounding import (
    EvidencePage,
    build_messages,
    citation_keys,
    load_day6_cases,
    parse_grounded_answer,
    required_fact_coverage,
    validate_citation_policy,
)
from evidencemm.hybrid_retrieval import RankedPage, fuse_rrf
from evidencemm.text_retrieval import BM25Index, load_corpus
from evidencemm.visual_corpus import load_visual_manifest
from evidencemm.visual_retrieval import (
    encode_query,
    load_model_and_processor,
    rank_scores,
    score_documents,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = yaml.safe_load(
        (ROOT / "configs/grounded_generation.yaml").read_text(
            encoding="utf-8"
        )
    )
    hybrid_config = yaml.safe_load(
        (ROOT / config["hybrid_config"]).read_text(
            encoding="utf-8"
        )
    )
    text_config = yaml.safe_load(
        (ROOT / hybrid_config["text_config"]).read_text(
            encoding="utf-8"
        )
    )
    visual_config = yaml.safe_load(
        (ROOT / hybrid_config["visual_config"]).read_text(
            encoding="utf-8"
        )
    )

    cases = load_day6_cases(ROOT / config["eval_cases"])
    text_documents = load_corpus(ROOT / config["text_corpus"])
    text_by_key = {
        (page.source_id, page.page_number): page
        for page in text_documents
    }

    visual_pages = load_visual_manifest(
        ROOT / config["visual_manifest"]
    )
    visual_by_key = {
        (page.source_id, page.page_number): page
        for page in visual_pages
    }

    text_index = BM25Index(
        text_documents,
        k1=float(text_config["k1"]),
        b=float(text_config["b"]),
    )
    visual_index = torch.load(
        ROOT / config["visual_index"],
        map_location="cpu",
        weights_only=False,
    )

    retrieval_model, retrieval_processor, retrieval_load_sec = (
        load_model_and_processor(
            model_name=visual_config["model_name"],
            max_num_visual_tokens=int(
                visual_config["max_num_visual_tokens"]
            ),
        )
    )

    torch.cuda.reset_peak_memory_stats()
    retrieval_started = time.perf_counter()
    selected_evidence = {}
    retrieval_rows = []

    component_top_k = int(hybrid_config["top_k"])
    evidence_top_k = int(config["evidence_top_k"])

    for case in cases:
        text_raw = text_index.search(
            case.question,
            top_k=component_top_k,
        )
        text_hits = [
            RankedPage(
                source_id=hit.source_id,
                page_number=hit.page_number,
                rank=hit.rank,
                raw_score=hit.score,
            )
            for hit in text_raw
        ]

        query_embedding, query_sec = encode_query(
            model=retrieval_model,
            processor=retrieval_processor,
            query=case.question,
        )
        vision_scores = score_documents(
            processor=retrieval_processor,
            query_embedding=query_embedding,
            document_embeddings=visual_index["embeddings"],
            device=retrieval_model.device,
        )
        vision_ranking = rank_scores(
            vision_scores,
            top_k=component_top_k,
        )

        vision_hits = []
        for rank, (doc_index, score) in enumerate(
            vision_ranking,
            start=1,
        ):
            page = visual_index["pages"][doc_index]
            vision_hits.append(
                RankedPage(
                    source_id=page["source_id"],
                    page_number=page["page_number"],
                    rank=rank,
                    raw_score=score,
                )
            )

        hybrid_hits = fuse_rrf(
            text_hits=text_hits,
            vision_hits=vision_hits,
            top_k=component_top_k,
            rrf_k=int(hybrid_config["rrf_k"]),
            text_weight=float(hybrid_config["text_weight"]),
            vision_weight=float(hybrid_config["vision_weight"]),
        )

        evidence_pages = []
        for hit in hybrid_hits[:evidence_top_k]:
            key = (hit.source_id, hit.page_number)
            text_page = text_by_key[key]
            visual_page = visual_by_key[key]
            image_path = ROOT / visual_page.image_path

            if sha256_file(image_path) != visual_page.image_sha256:
                raise SystemExit(
                    "visual evidence SHA256 drift: "
                    + visual_page.image_path
                )

            evidence_pages.append(
                EvidencePage(
                    source_id=hit.source_id,
                    page_number=hit.page_number,
                    image_path=str(image_path.resolve()),
                    text=text_page.text,
                    retrieval_rank=hit.rank,
                    rrf_score=hit.rrf_score,
                )
            )

        selected_evidence[case.case_id] = evidence_pages
        retrieval_rows.append(
            {
                "case_id": case.case_id,
                "query_encode_sec": query_sec,
                "hybrid_top5": [
                    hit.to_dict()
                    for hit in hybrid_hits
                ],
                "selected_evidence": [
                    page.to_dict()
                    for page in evidence_pages
                ],
            }
        )

        del query_embedding
        del vision_scores

    torch.cuda.synchronize()
    retrieval_sec = time.perf_counter() - retrieval_started
    retrieval_peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    del retrieval_model
    del retrieval_processor
    del visual_index
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    post_retrieval_allocated_mb = (
        torch.cuda.memory_allocated()
        / 1024
        / 1024
    )

    torch.cuda.reset_peak_memory_stats()

    generator, generator_processor, generator_load_sec = (
        load_generator(
            model_name=config["generator_model"],
            min_visual_tokens_per_image=int(
                config["generator_min_visual_tokens_per_image"]
            ),
            max_visual_tokens_per_image=int(
                config["generator_max_visual_tokens_per_image"]
            ),
            attn_implementation=config["attn_implementation"],
        )
    )

    rows = []
    structured_ok = 0
    answerability_ok = 0
    citation_policy_ok = 0
    answerable_gold_hits = 0
    answerable_citation_precision_sum = 0.0
    fact_coverage_sum = 0.0
    abstention_ok = 0
    pass_count = 0

    answerable_count = sum(
        case.expected_answerable
        for case in cases
    )
    abstention_count = len(cases) - answerable_count

    generation_started = time.perf_counter()

    for case in cases:
        evidence = selected_evidence[case.case_id]
        messages = build_messages(
            question=case.question,
            evidence=evidence,
        )

        raw_output, generation_sec = generate_grounded(
            model=generator,
            processor=generator_processor,
            messages=messages,
            max_new_tokens=int(config["max_new_tokens"]),
        )

        parse_error = None
        parsed = None
        citation_errors = []
        citation_valid = False
        predicted_answerable = None
        answerability_correct = False
        gold_hit = False
        citation_precision = 0.0
        fact_coverage = 0.0
        case_pass = False

        try:
            parsed = parse_grounded_answer(raw_output)
            structured_ok += 1

            predicted_answerable = not parsed.abstain
            answerability_correct = (
                predicted_answerable
                == case.expected_answerable
            )
            if answerability_correct:
                answerability_ok += 1

            citation_valid, citation_errors = (
                validate_citation_policy(
                    parsed,
                    evidence,
                )
            )
            if citation_valid:
                citation_policy_ok += 1

            cited = citation_keys(parsed.citations)
            gold = {
                (case.source_id, page_number)
                for page_number in case.gold_pages
            }

            if case.expected_answerable:
                gold_hit = bool(cited & gold)
                if gold_hit:
                    answerable_gold_hits += 1

                if cited:
                    citation_precision = (
                        len(cited & gold)
                        / len(cited)
                    )
                answerable_citation_precision_sum += (
                    citation_precision
                )

                fact_coverage = required_fact_coverage(
                    parsed.answer,
                    case.required_fact_groups,
                )
                fact_coverage_sum += fact_coverage

                case_pass = (
                    not parsed.abstain
                    and answerability_correct
                    and citation_valid
                    and gold_hit
                    and fact_coverage == 1.0
                )
            else:
                abstention_correct = (
                    parsed.abstain
                    and not parsed.citations
                )
                if abstention_correct:
                    abstention_ok += 1

                fact_coverage = 1.0
                case_pass = (
                    answerability_correct
                    and citation_valid
                    and abstention_correct
                )

        except Exception as exc:
            parse_error = (
                f"{type(exc).__name__}: {exc}"
            )

        if case_pass:
            pass_count += 1

        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_answerable": case.expected_answerable,
                "gold_pages": case.gold_pages,
                "evidence": [
                    page.to_dict()
                    for page in evidence
                ],
                "raw_output": raw_output,
                "parsed_output": (
                    parsed.model_dump()
                    if parsed is not None
                    else None
                ),
                "parse_error": parse_error,
                "predicted_answerable": predicted_answerable,
                "answerability_correct": answerability_correct,
                "citation_policy_valid": citation_valid,
                "citation_errors": citation_errors,
                "citation_gold_hit": gold_hit,
                "citation_precision": citation_precision,
                "required_fact_coverage": fact_coverage,
                "generation_sec": generation_sec,
                "case_pass": case_pass,
            }
        )

    torch.cuda.synchronize()
    generation_total_sec = (
        time.perf_counter()
        - generation_started
    )
    generator_peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    n = len(cases)

    result = {
        "mode": "hybrid_retrieval_grounded_qwen3vl",
        "generator_model": config["generator_model"],
        "cases": n,
        "answerable_cases": answerable_count,
        "abstention_cases": abstention_count,
        "evidence_top_k": evidence_top_k,
        "structured_output_rate": structured_ok / n,
        "answerability_accuracy": answerability_ok / n,
        "citation_policy_valid_rate": citation_policy_ok / n,
        "citation_gold_hit_rate": (
            answerable_gold_hits / answerable_count
            if answerable_count
            else 0.0
        ),
        "mean_citation_precision": (
            answerable_citation_precision_sum / answerable_count
            if answerable_count
            else 0.0
        ),
        "mean_required_fact_coverage": (
            fact_coverage_sum / answerable_count
            if answerable_count
            else 0.0
        ),
        "abstention_accuracy": (
            abstention_ok / abstention_count
            if abstention_count
            else 0.0
        ),
        "end_to_end_pass_rate": pass_count / n,
        "retrieval": {
            "model": visual_config["model_name"],
            "model_load_sec": retrieval_load_sec,
            "retrieval_sec": retrieval_sec,
            "peak_allocated_gpu_mb": retrieval_peak_mb,
            "post_unload_allocated_gpu_mb":
                post_retrieval_allocated_mb,
            "cases": retrieval_rows,
        },
        "generation": {
            "model_load_sec": generator_load_sec,
            "generation_total_sec": generation_total_sec,
            "peak_allocated_gpu_mb": generator_peak_mb,
        },
        "case_results": rows,
        "scope_note": (
            "Day 6 smoke baseline over two answerable PDF cases "
            "and one controlled unsupported PDF query. "
            "Deterministic validators measure structured output, "
            "citation binding, required fact coverage, and "
            "abstention; this is not a headline answer-quality "
            "benchmark."
        ),
    }

    output = ROOT / config["report_file"]
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
