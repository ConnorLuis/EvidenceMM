#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "data/protocol/day35_release_manifest.json"

DAY34_FINAL = "c80525f18062ada2613518f16f47868552ede0ea"

MILESTONE_COMMITS = {
    "day22_protocol_freeze": "5fcbc60bda246604a8405354320a23704a8cf70d",
    "day23_pilot_freeze": "64901f7093e46d21104e65b845ace4c2c104ca83",
    "day24_target_collection": "8c1035b6592488f6cf2a7cf4bbdddc86a62b6394",
    "day25_gripper_collection": "2eb16ae1fb9418af0a7c712dc321b69fd3f0ed42",
    "day26_trajectory_collection": "ba7669b5503cab13524c4ab1e4a5ad68c503abe2",
    "day27_evidence_challenge": "eaa29a3ebc9f41fa26ffa6de3291c6a28d93a4cd",
    "day28_raw_audit": "48ee78f7ba07f6e053e1581edff3d271e964c581",
    "day29_ground_truth": "98dfa730ae87193b907a818d75e50020daf5e567",
    "day30_group_split": "21a68f1df2a6f0770b3db4b5ad99fa6f16d481b1",
    "day31_baseline": "eb423b152555533e577315667e85067dd47069b7",
    "day32_calibration": "2b2b71c4489021f5637e8a4a5f6e6b3df36b0aa1",
    "day33_heldout": "a8a8b796eecdab6118c9ad637c41f7c2b987304d",
    "day34_metrics": DAY34_FINAL,
}

FROZEN_DAY34_HASHES = {
    "data/eval/day34_final_metrics_report.json":
        "536d49aff909545310867294a2cdd2f2626498f6d0220a459d07d873e31845e9",
    "data/eval/day34_error_analysis.json":
        "f32f44492d26cf344e7e056a4d8edef94a3af7f85743dd0dc959e8e323cd4e55",
    "data/eval/day34_per_case_analysis.jsonl":
        "130beb67cd9df54faa7b621338c60c5575d26df619d9f55b41dd7184c5853bf6",
    "data/eval/day34_development_e2e_profile.json":
        "a97d4d3e3a8c709c9b9d11999fd221cba91446438daeb2cf7e470d35b07fa208",
    "data/eval/day34_efficiency_report.json":
        "2002015a54afc95d8aa181dcc2c14ed394b193d4d45586b5673d7a7d35c6873d",
    "data/protocol/day34_metrics_freeze_receipt.json":
        "0c7fe40e62a4086b7de8d86b4c13ebc273cf82e6f69963dee2da3bff558b9c2b",
}

REQUIRED_README_TOKENS = (
    "90 canonical episodes in 15 pair groups",
    "development: **60 episodes / 10 pair groups**",
    "held-out: **30 episodes / 5 pair groups**",
    "Answerable 3-class Macro-F1",
    "0.1672",
    "gripper recall is **0.0**",
    "Structured-output parse rate",
    "Day35  project/release/documentation closure",
    "interview-ready research prototype",
)

FORBIDDEN_STALE_README_TOKENS = (
    "Not completed and not claimed",
    "robot failed-grasp root-cause diagnosis;",
    "The remaining flagship task is real robot-operation failure diagnosis",
    "is now moving from module validation toward the flagship robot-failure application",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_ancestor(commit: str, head: str = "HEAD") -> None:
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, head],
        cwd=ROOT,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"required ancestor missing: {commit}")


def validate_manifest() -> list[str]:
    errors: list[str] = []
    if not MANIFEST_PATH.exists():
        return ["release manifest missing"]

    manifest = read_json(MANIFEST_PATH)
    if manifest.get("schema_version") != "evidencemm_day35_release_manifest_v1":
        errors.append("manifest schema mismatch")
    if manifest.get("status") != "interview_ready_research_prototype_frozen_day35":
        errors.append("manifest status mismatch")
    if manifest.get("parent_day34_commit") != DAY34_FINAL:
        errors.append("manifest Day34 parent mismatch")

    if manifest.get("benchmark_summary") != {
        "canonical_episode_count": 90,
        "pair_group_count": 15,
        "development_episode_count": 60,
        "held_out_episode_count": 30,
        "held_out_final_evaluation_count_consumed": 1,
    }:
        errors.append("manifest benchmark summary mismatch")

    release_files = manifest.get("release_files")
    if not isinstance(release_files, dict):
        errors.append("manifest release_files missing")
    else:
        for rel, expected in release_files.items():
            path = ROOT / rel
            if not path.exists():
                errors.append(f"release file missing: {rel}")
                continue
            actual = sha256(path)
            if actual != expected:
                errors.append(
                    f"release file SHA mismatch: {rel} "
                    f"expected={expected} actual={actual}"
                )

    frozen = manifest.get("frozen_day34_artifacts")
    if frozen != FROZEN_DAY34_HASHES:
        errors.append("manifest frozen Day34 hashes mismatch")

    return errors


def validate_readme() -> list[str]:
    errors: list[str] = []
    path = ROOT / "README.md"
    if not path.exists():
        return ["README.md missing"]
    text = path.read_text(encoding="utf-8")

    for token in REQUIRED_README_TOKENS:
        if token not in text:
            errors.append(f"README required token missing: {token!r}")
    for token in FORBIDDEN_STALE_README_TOKENS:
        if token in text:
            errors.append(f"README stale token remains: {token!r}")

    if "high-accuracy root-cause classifier" not in text:
        errors.append("README non-claim language missing")
    if "no post-held-out tuning" not in text:
        errors.append("README post-heldout integrity statement missing")

    return errors


def validate_frozen_chain() -> list[str]:
    errors: list[str] = []

    if git("branch", "--show-current") != "master":
        errors.append("Day35 must run on master")

    for name, commit in MILESTONE_COMMITS.items():
        try:
            require_ancestor(commit)
        except RuntimeError:
            errors.append(f"milestone not ancestor: {name}={commit}")

    for rel, expected in FROZEN_DAY34_HASHES.items():
        path = ROOT / rel
        if not path.exists():
            errors.append(f"frozen Day34 artifact missing: {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(
                f"frozen Day34 artifact changed: {rel} "
                f"expected={expected} actual={actual}"
            )

    return errors


def cmd_validate() -> None:
    errors = []
    errors.extend(validate_frozen_chain())
    errors.extend(validate_manifest())
    errors.extend(validate_readme())

    manifest = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}

    print("===== DAY35 RELEASE VALIDATION =====")
    print("branch =", git("branch", "--show-current"))
    print("head =", git("rev-parse", "HEAD"))
    print("parent_day34_commit =", DAY34_FINAL)
    print("release_file_count =", len(manifest.get("release_files", {})))
    print("milestone_count =", len(MILESTONE_COMMITS))
    print("frozen_day34_artifact_count =", len(FROZEN_DAY34_HASHES))
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("DAY35 RELEASE VALIDATION: PASS")


def cmd_audit() -> None:
    cmd_validate()

    errors: list[str] = []
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^")

    if parent != DAY34_FINAL:
        errors.append(
            f"Day35 final commit must directly parent Day34: "
            f"expected={DAY34_FINAL} actual={parent}"
        )

    manifest = read_json(MANIFEST_PATH)
    for rel in manifest["release_files"]:
        try:
            git("rev-parse", f"HEAD:{rel}")
        except subprocess.CalledProcessError:
            errors.append(f"release file not tracked in final commit: {rel}")

    try:
        git("rev-parse", "HEAD:data/protocol/day35_release_manifest.json")
    except subprocess.CalledProcessError:
        errors.append("release manifest not tracked")

    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        errors.append("worktree is not clean")

    print("===== DAY35 FINAL PROJECT AUDIT =====")
    print("final_commit =", head)
    print("parent_day34_commit =", parent)
    print("worktree_clean =", not bool(status))
    print("project_status =", manifest["status"])
    print("errors =", errors)
    if errors:
        raise SystemExit(1)

    print("DAY35 PROJECT CLOSURE AUDIT: PASS")
    print("EVIDENCEMM: STAGE COMPLETE / INTERVIEW READY / FROZEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.cmd == "validate":
        cmd_validate()
    elif args.cmd == "audit":
        cmd_audit()


if __name__ == "__main__":
    main()
