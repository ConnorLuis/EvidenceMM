#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DAY35_FINAL = "4ad548dae04f965c7bf5d1becac1dabb63736cd3"
DAY35_MANIFEST_BLOB = "3637b9f680526dba1be2700e465aca8edbd53fab"

PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / "constraints.txt"
SNAPSHOT = ROOT / "data/protocol/day35_environment_snapshot.json"
WORKFLOW = ROOT / ".github/workflows/cpu-static-audit.yml"

DAY35_RELEASE_MANIFEST = ROOT / "data/protocol/day35_release_manifest.json"

DAY35_RELEASE_FILES = (
    "README.md",
    "docs/day35_benchmark_card.md",
    "docs/day35_reproducibility.md",
    "docs/day35_limitations.md",
    "docs/day35_project_closure.md",
    "docs/day35_release_notes.md",
    "scripts/day35_release_audit.py",
    "tests/test_day35_release.py",
    "data/protocol/day35_release_manifest.json",
)

REQUIRED_WORKFLOW_TOKENS = (
    "runs-on: ubuntu-latest",
    'python-version: "3.11.15"',
    "fetch-depth: 0",
    "python -m compileall -q src scripts tests",
    "python scripts/day35_release_audit.py validate",
    "python scripts/day34_metrics_error_efficiency.py audit",
    "python scripts/release_hardening_audit.py validate",
    "tests/test_day31_root_cause_baseline.py",
    "tests/test_release_hardening.py",
)

FORBIDDEN_WORKFLOW_TOKENS = (
    "Qwen3VLForConditionalGeneration",
    "from_pretrained",
    "cuda",
    "nvidia",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_ancestor(commit: str) -> None:
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"required ancestor missing: {commit}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def requirement_name(requirement: str) -> str:
    left = requirement.split("@", 1)[0].strip()
    for token in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        if token in left:
            left = left.split(token, 1)[0].strip()
    return normalize_name(left)


def declared_dependency_names() -> set[str]:
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    names: set[str] = set()
    for req in payload["project"].get("dependencies", []):
        names.add(requirement_name(req))
    for req in payload["project"].get("optional-dependencies", {}).get("dev", []):
        names.add(requirement_name(req))
    for req in payload.get("build-system", {}).get("requires", []):
        names.add(requirement_name(req))
    return names


def parse_constraints() -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = requirement_name(line)
        if name in out:
            raise RuntimeError(f"duplicate constraint for {name}")
        out[name] = line
    return out


def validate_day35_immutability() -> list[str]:
    errors: list[str] = []
    try:
        require_ancestor(DAY35_FINAL)
    except RuntimeError as exc:
        errors.append(str(exc))
        return errors

    try:
        current_manifest_blob = git(
            "rev-parse", "HEAD:data/protocol/day35_release_manifest.json"
        )
        if current_manifest_blob != DAY35_MANIFEST_BLOB:
            errors.append("Day35 release manifest changed after freeze")
    except subprocess.CalledProcessError:
        errors.append("Day35 release manifest missing")
        return errors

    for rel in DAY35_RELEASE_FILES:
        try:
            frozen_blob = git("rev-parse", f"{DAY35_FINAL}:{rel}")
            current_blob = git("rev-parse", f"HEAD:{rel}")
        except subprocess.CalledProcessError:
            errors.append(f"Day35 frozen release file missing: {rel}")
            continue
        if frozen_blob != current_blob:
            errors.append(f"Day35 frozen release file changed: {rel}")
    return errors


def validate_constraints() -> list[str]:
    errors: list[str] = []
    if not CONSTRAINTS.exists():
        return ["constraints.txt missing"]
    if not SNAPSHOT.exists():
        return ["environment snapshot missing"]

    constraints = parse_constraints()
    declared = declared_dependency_names()
    missing = sorted(declared - set(constraints))
    if missing:
        errors.append(f"unfrozen declared dependencies: {missing}")

    for name in sorted(declared):
        line = constraints.get(name, "")
        if "@" in line:
            if "@c23838d920a7c426ee297034211cff2f55da65dc" not in line:
                errors.append(f"direct URL not immutable: {name}")
        elif not re.fullmatch(r"[a-z0-9_.-]+==[^\\s]+", line):
            errors.append(f"constraint is not exact pin: {line!r}")

    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != "evidencemm_day35_environment_snapshot_v1":
        errors.append("environment snapshot schema mismatch")
    if snapshot.get("source_day35_commit") != DAY35_FINAL:
        errors.append("environment snapshot Day35 source mismatch")
    if snapshot.get("python_version") != "3.11.15":
        errors.append("environment snapshot Python mismatch")

    packages = snapshot.get("packages", {})
    runtime_builds = snapshot.get("runtime_builds", {})

    if runtime_builds.get("torch_runtime_version") != "2.11.0+cu130":
        errors.append(
            "environment snapshot torch runtime mismatch: "
            f"actual={runtime_builds.get('torch_runtime_version')}"
        )
    if packages.get("torch", {}).get("installed_version") != "2.11.0":
        errors.append(
            "environment snapshot torch distribution mismatch: "
            f"actual={packages.get('torch', {}).get('installed_version')}"
        )
    if packages.get("transformers", {}).get("installed_version") != "5.15.0":
        errors.append(
            "environment snapshot transformers mismatch: "
            f"actual={packages.get('transformers', {}).get('installed_version')}"
        )

    if "torchvision" not in packages:
        errors.append("environment snapshot torchvision missing")

    if snapshot.get("constraints_sha256") != sha256(CONSTRAINTS):
        errors.append("environment snapshot constraints SHA mismatch")

    snapshot_names = set(packages)
    missing_snapshot = sorted(declared - snapshot_names)
    if missing_snapshot:
        errors.append(
            f"declared dependencies absent from environment snapshot: "
            f"{missing_snapshot}"
        )

    return errors


def validate_workflow() -> list[str]:
    errors: list[str] = []
    if not WORKFLOW.exists():
        return ["CPU static audit workflow missing"]
    text = WORKFLOW.read_text(encoding="utf-8")

    for token in REQUIRED_WORKFLOW_TOKENS:
        if token not in text:
            errors.append(f"workflow required token missing: {token!r}")
    lowered = text.lower()
    for token in FORBIDDEN_WORKFLOW_TOKENS:
        if token.lower() in lowered:
            errors.append(f"workflow unexpectedly references GPU/model load: {token}")

    if "contents: read" not in text:
        errors.append("workflow permissions are not read-only")
    if "timeout-minutes: 10" not in text:
        errors.append("workflow timeout missing")

    return errors


def validate() -> list[str]:
    errors: list[str] = []
    if git("branch", "--show-current") != "master":
        errors.append("release hardening must run on master")
    errors.extend(validate_day35_immutability())
    errors.extend(validate_constraints())
    errors.extend(validate_workflow())
    return errors


def cmd_validate() -> None:
    errors = validate()
    print("===== EVIDENCEMM POST-DAY35 RELEASE HARDENING VALIDATION =====")
    print("head =", git("rev-parse", "HEAD"))
    print("day35_frozen_ancestor =", DAY35_FINAL)
    print("constraints_sha256 =", sha256(CONSTRAINTS) if CONSTRAINTS.exists() else None)
    print("workflow =", WORKFLOW.relative_to(ROOT))
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("RELEASE HARDENING VALIDATION: PASS")


def cmd_audit() -> None:
    cmd_validate()
    errors: list[str] = []

    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if status:
        errors.append("worktree is not clean")

    print("===== EVIDENCEMM POST-DAY35 RELEASE HARDENING AUDIT =====")
    print("final_head =", git("rev-parse", "HEAD"))
    print("day35_release_files_unchanged = true")
    print("cpu_static_ci_present = true")
    print("key_dependency_constraints_present = true")
    print("worktree_clean =", not bool(status))
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("RELEASE HARDENING AUDIT: PASS")
    print("EVIDENCEMM REPOSITORY CLOSURE: FULL PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("audit")
    args = parser.parse_args()
    if args.cmd == "validate":
        cmd_validate()
    else:
        cmd_audit()


if __name__ == "__main__":
    main()
