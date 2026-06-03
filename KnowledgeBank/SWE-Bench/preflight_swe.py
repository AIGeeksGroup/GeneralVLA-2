#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
from pathlib import Path

from datasets import DownloadConfig, load_dataset


PACKAGE_REQUIREMENTS = {
    "baseline": ["datasets", "litellm"],
    "cm": ["datasets", "litellm"],
}


def _detect_provider(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("vertex_ai/"):
        return "vertex_ai"
    if any(token in lowered for token in ["claude", "sonnet", "opus", "anthropic"]):
        return "anthropic"
    if any(token in lowered for token in ["gemini", "google"]):
        return "google"
    if "deepseek" in lowered:
        return "deepseek"
    if any(token in lowered for token in ["gpt", "openai"]):
        return "openai"
    return "unknown"


def _required_envs(mode: str, provider: str) -> list[str]:
    envs: list[str] = []
    if provider == "vertex_ai":
        envs.extend(["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"])
    elif provider == "anthropic":
        envs.append("ANTHROPIC_API_KEY")
    elif provider == "google":
        envs.append("GEMINI_API_KEY")
        envs.append("GOOGLE_API_KEY")
    elif provider == "deepseek":
        envs.append("DEEPSEEK_API_KEY")
    elif provider == "openai":
        envs.append("OPENAI_API_KEY")
    return envs


def _check_packages(mode: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for package_name in PACKAGE_REQUIREMENTS[mode]:
        try:
            importlib.import_module(package_name)
            results[package_name] = "ok"
        except Exception as exc:
            results[package_name] = f"missing: {type(exc).__name__}: {exc}"
    return results


def _check_dataset(subset: str, split: str, *, local_only: bool) -> dict[str, str | int]:
    try:
        dataset = load_dataset(
            subset,
            split=split,
            download_config=DownloadConfig(local_files_only=local_only),
        )
        first_instance = dataset[0]["instance_id"] if len(dataset) else ""
        return {"status": "ok", "rows": len(dataset), "first_instance_id": first_instance}
    except Exception as exc:
        return {"status": f"error: {type(exc).__name__}: {exc}"}


def _check_env_vars(required_envs: list[str]) -> dict[str, str]:
    return {env_name: ("set" if os.getenv(env_name) else "unset") for env_name in required_envs}


def _check_system_tools() -> dict[str, str]:
    return {tool: ("ok" if shutil.which(tool) else "missing") for tool in ["docker", "git", "python3", "gcloud"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the local machine is ready for SWE-Bench runs.")
    parser.add_argument("--mode", choices=["baseline", "cm"], default="cm", help="Which experiment path to validate")
    parser.add_argument("--model", default="deepseek/deepseek-chat", help="Model name used for the run")
    parser.add_argument("--subset", default="princeton-nlp/SWE-Bench_Verified", help="Dataset name or path")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--local-dataset-only", action="store_true", help="Require the dataset to be already cached locally")
    parser.add_argument("--memory-root", type=Path, help="Optional memory root to inspect")
    args = parser.parse_args()

    provider = _detect_provider(args.model)
    payload = {
        "mode": args.mode,
        "model": args.model,
        "provider": provider,
        "packages": _check_packages(args.mode),
        "env_vars": _check_env_vars(_required_envs(args.mode, provider)),
        "system_tools": _check_system_tools(),
        "dataset": _check_dataset(args.subset, args.split, local_only=args.local_dataset_only),
    }
    if args.memory_root:
        payload["memory_root_exists"] = args.memory_root.exists()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
