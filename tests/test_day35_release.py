from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "day35_release",
    ROOT / "scripts/day35_release_audit.py",
)
assert spec is not None and spec.loader is not None
day35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day35)


def test_day34_frozen_artifacts_have_expected_hashes() -> None:
    for rel, expected in day35.FROZEN_DAY34_HASHES.items():
        assert day35.sha256(ROOT / rel) == expected


def test_all_milestone_commits_are_ancestors() -> None:
    for commit in day35.MILESTONE_COMMITS.values():
        day35.require_ancestor(commit)


def test_release_manifest_contract() -> None:
    manifest = json.loads(
        day35.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "evidencemm_day35_release_manifest_v1"
    assert manifest["status"] == "interview_ready_research_prototype_frozen_day35"
    assert manifest["parent_day34_commit"] == day35.DAY34_FINAL


def test_release_manifest_benchmark_counts() -> None:
    manifest = json.loads(
        day35.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert manifest["benchmark_summary"] == {
        "canonical_episode_count": 90,
        "pair_group_count": 15,
        "development_episode_count": 60,
        "held_out_episode_count": 30,
        "held_out_final_evaluation_count_consumed": 1,
    }


def test_release_files_match_manifest_hashes() -> None:
    manifest = json.loads(
        day35.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    for rel, expected in manifest["release_files"].items():
        assert day35.sha256(ROOT / rel) == expected


def test_readme_has_final_status_and_no_stale_flagship_claim() -> None:
    errors = day35.validate_readme()
    assert errors == []


def test_readme_exposes_frozen_heldout_result() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "0.1672" in text
    assert "0.1157" in text
    assert "0.8333" in text
    assert "gripper recall is **0.0**" in text


def test_benchmark_card_preserves_negative_result() -> None:
    text = (ROOT / "docs/day35_benchmark_card.md").read_text(encoding="utf-8")
    assert "predicted gripper zero times" in text
    assert "No held-out clean control is correctly predicted" not in text
    assert "0.167224" in text
    assert "0.0000" in text


def test_limitations_do_not_overclaim_accuracy() -> None:
    text = (ROOT / "docs/day35_limitations.md").read_text(encoding="utf-8")
    assert "does not demonstrate that physical root-cause diagnosis is solved" in text
    assert "high-accuracy" not in text


def test_reproducibility_lists_final_audit_chain() -> None:
    text = (ROOT / "docs/day35_reproducibility.md").read_text(encoding="utf-8")
    for day in ("day31", "day32", "day33", "day34", "day35"):
        assert f"scripts/{day}" in text


def test_release_notes_mark_research_not_production() -> None:
    text = (ROOT / "docs/day35_release_notes.md").read_text(encoding="utf-8")
    assert "staged research-project closure" in text
    assert "not a production release" in text
