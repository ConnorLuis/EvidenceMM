# EvidenceMM

EvidenceMM is a traceable multimodal RAG system for complex documents and robot-operation videos.

## Current scope

Day1 establishes a reproducible direct-VLM baseline before retrieval is introduced.

Inputs planned for the full system:
- PDF manuals and page images
- standalone images and figures
- wrist/front robot videos
- robot joint state and action streams

Outputs planned for the full system:
- answer or explicit abstention
- page / timestamp / frame / camera evidence
- optional normalized region (`bbox`) evidence

## Day1 non-goals

- no LangGraph
- no FastAPI service
- no Qdrant
- no ColPali / ColQwen retrieval
- no OCR pipeline
- no reranker
- no LoRA / QLoRA

## Day1 commands

```bash
python scripts/check_env.py
python scripts/validate_eval_cases.py
pytest -q
python scripts/run_vlm_baseline.py --image /absolute/path/to/image.jpg --question "图中机械臂处于什么状态？"
```

See `docs/task_definition.md` for the project contract.
