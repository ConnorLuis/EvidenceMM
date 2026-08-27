from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "hardening",
    ROOT / "scripts/release_hardening_audit.py",
)
assert spec is not None and spec.loader is not None
hardening = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hardening)


def test_day35_release_commit_is_ancestor() -> None:
    hardening.require_ancestor(hardening.DAY35_FINAL)


def test_day35_release_files_are_immutable() -> None:
    assert hardening.validate_day35_immutability() == []


def test_constraints_cover_every_declared_dependency() -> None:
    constraints = hardening.parse_constraints()
    declared = hardening.declared_dependency_names()
    assert declared.issubset(set(constraints))


def test_torch_and_torchvision_are_frozen() -> None:
    constraints = hardening.parse_constraints()
    assert constraints["torch"].startswith("torch==")
    assert constraints["torchvision"].startswith("torchvision==")


def test_environment_snapshot_matches_frozen_runtime_evidence() -> None:
    snapshot = json.loads(
        hardening.SNAPSHOT.read_text(encoding="utf-8")
    )
    assert snapshot["python_version"] == "3.11.15"
    assert snapshot["packages"]["torch"]["installed_version"] == "2.11.0"
    assert snapshot["runtime_builds"]["torch_runtime_version"] == "2.11.0+cu130"
    assert snapshot["packages"]["transformers"]["installed_version"] == "5.15.0"
    assert "torchvision" in snapshot["packages"]


def test_constraints_hash_is_bound_into_snapshot() -> None:
    snapshot = json.loads(
        hardening.SNAPSHOT.read_text(encoding="utf-8")
    )
    assert snapshot["constraints_sha256"] == hardening.sha256(
        hardening.CONSTRAINTS
    )


def test_cpu_workflow_does_not_load_qwen_or_cuda() -> None:
    text = hardening.WORKFLOW.read_text(encoding="utf-8").lower()
    assert "qwen3vlforconditionalgeneration" not in text
    assert "from_pretrained" not in text
    assert "cuda" not in text
    assert "nvidia" not in text


def test_cpu_workflow_runs_static_recomputation_and_audits() -> None:
    text = hardening.WORKFLOW.read_text(encoding="utf-8")
    assert "python -m compileall -q src scripts tests" in text
    assert "python scripts/day34_metrics_error_efficiency.py audit" in text
    assert "python scripts/day35_release_audit.py validate" in text
    assert "python scripts/release_hardening_audit.py validate" in text

def test_cpu_workflow_fetches_full_history_for_ancestry_audits() -> None:
    text = hardening.WORKFLOW.read_text(encoding="utf-8")
    assert "fetch-depth: 0" in text
