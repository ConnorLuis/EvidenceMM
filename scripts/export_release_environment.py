#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / "constraints.txt"
SNAPSHOT = ROOT / "data/protocol/day35_environment_snapshot.json"

EXPECTED_PYTHON = "3.11.15"
EXPECTED_TORCH_FULL = "2.11.0+cu130"
EXPECTED_TRANSFORMERS = "5.15.0"

EXTRA_DISTRIBUTIONS = ("pytest",)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def requirement_name(requirement: str) -> str:
    left = requirement.split("@", 1)[0].strip()
    for token in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        if token in left:
            left = left.split(token, 1)[0].strip()
    return normalize_name(left)


def project_requirements() -> tuple[list[str], list[str], list[str]]:
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = payload["project"]
    runtime = list(project.get("dependencies", []))
    dev = list(project.get("optional-dependencies", {}).get("dev", []))
    build = list(payload.get("build-system", {}).get("requires", []))
    return runtime, dev, build


def installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"required project distribution is not installed: {name}"
        ) from exc


def pin_for_requirement(requirement: str) -> tuple[str, dict[str, Any]]:
    name = requirement_name(requirement)
    if "@" in requirement:
        # Preserve exact direct-URL requirement. The existing pyproject already
        # pins colpali-engine to an immutable Git commit.
        installed = installed_version(name)
        return requirement.strip(), {
            "name": name,
            "installed_version": installed,
            "constraint": requirement.strip(),
            "source": "direct_url_requirement",
        }

    installed = installed_version(name)
    constraint = f"{name}=={installed}"
    return constraint, {
        "name": name,
        "installed_version": installed,
        "constraint": constraint,
        "source": "resolved_distribution_version",
    }


def cuda_runtime_version() -> str | None:
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def gpu_name() -> str | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    runtime, dev, build = project_requirements()
    requirements = runtime + dev + build

    # Deduplicate by normalized package name, while preserving the strongest
    # project-declared direct URL if present.
    by_name: dict[str, str] = {}
    for req in requirements:
        name = requirement_name(req)
        if name in by_name and "@" not in req:
            continue
        by_name[name] = req

    for name in EXTRA_DISTRIBUTIONS:
        by_name.setdefault(normalize_name(name), name)

    pins: list[str] = []
    packages: dict[str, Any] = {}
    for name in sorted(by_name):
        constraint, record = pin_for_requirement(by_name[name])
        pins.append(constraint)
        packages[name] = record

    python_version = platform.python_version()
    torch_distribution_version = packages["torch"]["installed_version"]
    transformers_version = packages["transformers"]["installed_version"]

    try:
        import torch
        torch_runtime_version = torch.__version__
    except Exception as exc:
        raise RuntimeError("torch import failed while exporting environment") from exc

    if python_version != EXPECTED_PYTHON:
        raise RuntimeError(
            f"environment Python drift: expected {EXPECTED_PYTHON}, got {python_version}"
        )
    if torch_runtime_version != EXPECTED_TORCH_FULL:
        raise RuntimeError(
            "environment torch runtime drift: "
            f"expected {EXPECTED_TORCH_FULL}, got {torch_runtime_version}"
        )
    if transformers_version != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            "environment transformers drift: "
            f"expected {EXPECTED_TRANSFORMERS}, got {transformers_version}"
        )

    header = [
        "# EvidenceMM post-Day35 resolved environment constraints",
        "# Generated from the successful frozen benchmark environment.",
        "# CUDA-local package versions are intentionally preserved exactly.",
        "# GitHub Actions CPU CI installs only pytest under these constraints;",
        "# it does not install or load Qwen3-VL / torch runtime dependencies.",
        "",
    ]
    constraints_text = "\n".join(header + pins) + "\n"
    constraints_sha = sha256_bytes(constraints_text.encode("utf-8"))

    snapshot = {
        "schema_version": "evidencemm_day35_environment_snapshot_v1",
        "status": "successful_frozen_benchmark_environment_locked_post_day35",
        "source_day35_commit": "4ad548dae04f965c7bf5d1becac1dabb63736cd3",
        "captured_from_git_head": git_head(),
        "python_version": python_version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cuda_runtime_version": cuda_runtime_version(),
        "gpu_name": gpu_name(),
        "packages": packages,
        "runtime_builds": {
            "torch_runtime_version": torch_runtime_version,
            "torch_distribution_version": torch_distribution_version,
            "torchvision_distribution_version": packages["torchvision"]["installed_version"],
            "transformers_distribution_version": transformers_version,
        },
        "constraints_sha256": constraints_sha,
        "scope": {
            "runtime_direct_dependencies": sorted(
                requirement_name(req) for req in runtime
            ),
            "dev_direct_dependencies": sorted(
                requirement_name(req) for req in dev
            ),
            "build_direct_dependencies": sorted(
                requirement_name(req) for req in build
            ),
            "full_transitive_lock": False,
            "purpose": (
                "Freeze project-declared key dependency versions used by the "
                "successful Day31-Day35 environment."
            ),
        },
        "known_frozen_runtime_evidence": {
            "day34_python_version": "3.11.15",
            "day34_torch_version": "2.11.0+cu130",
            "day34_transformers_version": "5.15.0",
            "day34_cuda_runtime_version": "13.0",
            "day34_gpu_name": "NVIDIA GeForce RTX 4070 SUPER",
        },
    }

    print("python_version =", python_version)
    print("torch_runtime_version =", torch_runtime_version)
    print("torch_distribution_version =", torch_distribution_version)
    print("torchvision_distribution_version =", packages["torchvision"]["installed_version"])
    print("transformers_version =", transformers_version)
    print("constraint_count =", len(pins))
    print("constraints_sha256 =", constraints_sha)

    if args.write:
        CONSTRAINTS.write_text(
            constraints_text, encoding="utf-8", newline="\n"
        )
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print("wrote =", CONSTRAINTS.relative_to(ROOT))
        print("wrote =", SNAPSHOT.relative_to(ROOT))
    else:
        print(constraints_text)


if __name__ == "__main__":
    main()
