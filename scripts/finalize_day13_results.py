from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HYBRID_REPORT = (
    ROOT
    / "reports/day13_hybrid_retrieval.json"
)
RANKING_REPORT = (
    ROOT
    / "reports/day13_ranking_analysis.json"
)
HYBRID_DOC = (
    ROOT
    / "docs/day13_hybrid_retrieval.md"
)
RANKING_DOC = (
    ROOT
    / "docs/day13_ranking_analysis.md"
)

START = "<!-- DAY13_OBSERVED_START -->"
END = "<!-- DAY13_OBSERVED_END -->"


def replace_block(
    *,
    path: Path,
    body: str,
) -> None:
    text = path.read_text(
        encoding="utf-8"
    )
    if START not in text or END not in text:
        raise ValueError(
            f"result markers missing: {path}"
        )
    before, remainder = text.split(
        START,
        1,
    )
    _, after = remainder.split(
        END,
        1,
    )
    updated = (
        before
        + START
        + "\n"
        + body.rstrip()
        + "\n"
        + END
        + after
    )
    path.write_text(
        updated,
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    hybrid = json.loads(
        HYBRID_REPORT.read_text(
            encoding="utf-8"
        )
    )
    ranking = json.loads(
        RANKING_REPORT.read_text(
            encoding="utf-8"
        )
    )

    hybrid_body = f"""## Observed Day 13 result

```text
BM25:     {hybrid["bm25"]}
Dense:    {hybrid["dense"]}
Union:    {hybrid["candidate_union"]}
Reranker: {hybrid["reranker"]}
Delta reranker - BM25:
{hybrid["delta_reranker_minus_bm25"]}
```

This remains a {hybrid["case_count"]}-case smoke result.
The union candidate pool is variable-sized, so no same-budget recall
improvement claim is made.

Per-case first relevant ranks:

```text
""" + "\n".join(
        (
            f'{case["case_id"]}: '
            f'BM25={case["bm25_first_relevant_rank"]}, '
            f'Dense={case["dense_first_relevant_rank"]}, '
            f'UnionContainsGold={case["union_contains_gold"]}, '
            f'Reranker={case["reranker_first_relevant_rank"]}'
        )
        for case in hybrid["cases"]
    ) + "\n```"

    document = ranking["document"]
    robot = ranking["robot"]
    ranking_body = f"""## Observed ranking trace

Document case `{document["case_id"]}`, target page
`{document["target_page"]}`:

```text
BM25 rank: {document["bm25"]["rank"]}
Dense rank: {document["dense"]["rank"]}
Reranker rank: {document["reranker"]["rank"]}
```

Robot query `{robot["query"]}`:

```text
profile: {robot["profile"]}
top_score: {robot["top_score"]}
exact_top_score_tie_count: {robot["exact_top_score_tie_count"]}
exact_top_score_tied_frames: {robot["exact_top_score_tied_frames"]}
near_top_tolerance: {robot["near_top_tolerance"]}
near_top_score_count: {robot["near_top_score_count"]}
near_top_score_frames: {robot["near_top_score_frames"]}
selected_frames: {robot["selected_frames"]}
selection_rule: {robot["selection_rule"]}
```

The document trace explains exact BM25 term contributions, dense cosine
similarity, union provenance, and cross-encoder reranker rank. The robot trace
records the canonical signal profile and deterministic tie break. Neither trace
is a causal failure diagnosis.
"""

    replace_block(
        path=HYBRID_DOC,
        body=hybrid_body,
    )
    replace_block(
        path=RANKING_DOC,
        body=ranking_body,
    )

    print(
        "Day 13 observed results written into "
        "the two new Day 13 docs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
