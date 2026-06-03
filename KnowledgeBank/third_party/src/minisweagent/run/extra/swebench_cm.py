#!/usr/bin/env python3

"""Experimental consolidated-memory SWE-Bench entrypoint."""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.memory.instruction import FAILED_SI, SUCCESSFUL_SI
from minisweagent.memory_v2.consolidation import consolidate_active_records, resolve_candidate
from minisweagent.memory_v2.governance import run_budgeted_governance
from minisweagent.memory_v2.metrics import collect_memory_health
from minisweagent.memory_v2.retrieval import build_memory_block, rank_memory_records, select_top_memories
from minisweagent.memory_v2.schema import MemoryRecord, MemoryState, MemoryType
from minisweagent.memory_v2.signatures import build_signature
from minisweagent.memory_v2.store import JsonlMemoryStore
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.verifier.gemini_client import GeminiVerifierClient
from minisweagent.verifier.schema import VerificationLabel, VerificationResult
from minisweagent.verifier.scoring import verify_trajectory

_HELP_TEXT = """Run mini-SWE-agent on SWE-Bench instances with consolidated memory."""

app = typer.Typer(
    help="Experimental SWE-Bench runner for KnowledgeBank-CM.",
    rich_markup_mode="rich",
    add_completion=False,
)


def _load_baseline_runner():
    # Import lazily so the module stays importable in minimal test environments.
    from minisweagent.run.extra import swebench as baseline_swebench

    return baseline_swebench


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "default"


def _extract_trajectory_text(messages: list[dict]) -> str:
    chunks: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in content:
            continue
        if role == "user":
            if index == 1 and "<pr_description>" in content:
                continue
            if "Please always provide EXACTLY ONE action in triple backticks" in content:
                continue
            if content.startswith("<returncode>"):
                chunks.append(content[:4000])
                continue
        chunks.append(content)
    return "\n".join(chunks)


def _extract_markdown_field(block: str, label: str) -> str:
    pattern = re.compile(rf"^##\s*{re.escape(label)}\s+(.*?)(?=^##\s+\w|\Z)", re.MULTILINE | re.DOTALL)
    match = pattern.search(block)
    return match.group(1).strip() if match else ""


def parse_memory_items(raw_text: str) -> list[dict[str, str]]:
    blocks = re.findall(r"(?ms)^#\s*Memory Item.*?(?=^#\s*Memory Item|\Z)", raw_text.strip())
    items: list[dict[str, str]] = []
    for block in blocks:
        title = _extract_markdown_field(block, "Title")
        description = _extract_markdown_field(block, "Description")
        content = _extract_markdown_field(block, "Content")
        merged = "\n".join(part for part in [title, description, content] if part).strip()
        if merged:
            items.append(
                {
                    "title": title,
                    "description": description,
                    "content": content or description or title,
                    "full_text": merged,
                }
            )
    if items:
        return items

    cleaned = raw_text.strip()
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    bullet_candidates = re.split(r"(?:\s+-\s+|\s+\*\s+|\s+\d+\.\s+)", cleaned)
    bullet_candidates = [segment.strip(" -") for segment in bullet_candidates if segment.strip(" -")]
    if not bullet_candidates:
        bullet_candidates = [cleaned]

    for index, segment in enumerate(bullet_candidates[:3], start=1):
        short = segment[:240].strip()
        if len(short) < 20:
            continue
        title = short.split(".")[0][:80].strip() or f"Fallback Memory {index}"
        items.append(
            {
                "title": title,
                "description": short[:120],
                "content": short,
                "full_text": "\n".join([part for part in [title, short[:120], short] if part]).strip(),
            }
        )
    return items


def _normalize_dedup_key(memory_type: MemoryType, text: str) -> str:
    return f"{memory_type.value}:{build_signature(text, max_tokens=5)}"


def _infer_memory_type(item: dict[str, str], source_status: str) -> MemoryType:
    text = " ".join([item.get("title", ""), item.get("description", ""), item.get("content", "")]).lower()
    if source_status in {VerificationLabel.VERIFIED_FAIL.value, "fail"}:
        return MemoryType.FAILURE_AVOIDANCE
    if any(token in text for token in ["tool", "command", "rg ", "grep", "pytest", "git", "log", "traceback"]):
        return MemoryType.TOOL_USAGE
    if source_status not in {"success", VerificationLabel.VERIFIED_SUCCESS.value} or any(
        token in text for token in ["avoid", "failure", "mistake", "wrong"]
    ):
        return MemoryType.FAILURE_AVOIDANCE
    return MemoryType.PROCEDURAL_HINT


def build_candidate_records(
    *,
    task_id: str,
    query: str,
    raw_memory_text: str,
    source_status: str,
    verifier_result: VerificationResult | None = None,
    created_at: str | None = None,
) -> list[MemoryRecord]:
    created_at = created_at or _utcnow()
    items = parse_memory_items(raw_memory_text)
    records: list[MemoryRecord] = []
    for index, item in enumerate(items):
        record_status = verifier_result.label.value if verifier_result is not None else source_status
        memory_type = _infer_memory_type(item, record_status)
        confidence = verifier_result.confidence if verifier_result is not None else (0.75 if source_status == "success" else 0.45)
        quality_score = verifier_result.score if verifier_result is not None else (0.8 if source_status == "success" else 0.55)
        records.append(
            MemoryRecord(
                memory_id=f"{task_id}-{index}-{uuid.uuid4().hex[:8]}",
                task_id=task_id,
                query=query,
                content=item["full_text"],
                memory_type=memory_type,
                source_status=record_status,
                state=MemoryState.PROVISIONAL,
                confidence=confidence,
                quality_score=quality_score,
                created_at=created_at,
                last_used_at=None,
                use_count=0,
                dedup_key=_normalize_dedup_key(memory_type, item["full_text"]),
                supersedes=[],
                conflicts_with=[],
                embedding=None,
                verifier_score=verifier_result.score if verifier_result is not None else None,
                verifier_confidence=verifier_result.confidence if verifier_result is not None else None,
                verifier_label=verifier_result.label.value if verifier_result is not None else None,
                verifier_criteria=verifier_result.criteria_scores if verifier_result is not None else {},
                verifier_model=verifier_result.model_name if verifier_result is not None else None,
            )
        )
    return records


def _query_support_model(
    *,
    model_obj,
    prompt: str,
    system_instruction: str,
    temperature: float = 0.0,
    max_tokens: int = 384,
) -> str:
    response = model_obj.query(
        [
            {"role": "system", "content": system_instruction.strip()},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.get("content", "")


def _generate_memory_text(task: str, trajectory: str, model_obj, success: bool) -> str:
    system_instruction = SUCCESSFUL_SI if success else FAILED_SI
    return _query_support_model(
        model_obj=model_obj,
        prompt=f"**Query:** {task}\n\n**Trajectory:**\n{trajectory}",
        system_instruction=system_instruction,
        temperature=0.0,
        max_tokens=384,
    )


def _judge_status_with_model(task: str, trajectory: str, model_obj) -> bool:
    response = _query_support_model(
        model_obj=model_obj,
        prompt=(
            f"Task: {task}\n\nTrajectory:\n{trajectory}\n\n"
            "Did the agent successfully complete the task? Answer with 'success' or 'fail' only."
        ),
        system_instruction="You judge whether the agent successfully completed the task. Output only success or fail.",
        temperature=0.0,
        max_tokens=16,
    ).strip().lower()
    return "success" in response


class _ModelObjectVerifierClient:
    def __init__(self, model_obj):
        self.model_obj = model_obj

    def score_prompt(self, prompt: str) -> dict:
        response = self.model_obj.query(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict SWE-Bench trajectory verifier. "
                        "Return the requested score tag exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        return {"text": response.get("content", ""), "score": None, "used_logprobs": False}


def _verify_trajectory_with_model(
    *,
    task: str,
    trajectory: str,
    patch: str,
    exit_status: str,
    model_obj,
    verifier_mode: str,
    verifier_model: str,
    verifier_reps: int,
    verifier_success_threshold: float = 0.70,
    verifier_fail_threshold: float = 0.35,
) -> VerificationResult:
    if verifier_mode == "off":
        success = _judge_status_with_model(task, trajectory, model_obj)
        score = 0.8 if success else 0.2
        return VerificationResult.from_scores(
            criteria_scores={
                "root_cause": score,
                "code_review": score,
                "empirical_verification": score,
            },
            raw_outputs=["legacy judge fallback"],
            model_name=getattr(model_obj.config, "model_name", ""),
            n_reps=1,
            used_logprobs=False,
            exit_status=exit_status,
            success_threshold=verifier_success_threshold,
            fail_threshold=verifier_fail_threshold,
        )

    client = (
        _ModelObjectVerifierClient(model_obj)
        if verifier_mode == "text"
        else GeminiVerifierClient(model_name=verifier_model, mode=verifier_mode)
    )
    return verify_trajectory(
        task=task,
        trajectory=trajectory,
        patch=patch,
        exit_status=exit_status,
        client=client,
        n_reps=verifier_reps,
        model_name=verifier_model if verifier_mode != "text" else getattr(model_obj.config, "model_name", verifier_model),
        success_threshold=verifier_success_threshold,
        fail_threshold=verifier_fail_threshold,
    )


def _should_write_memory_for_verification(
    verifier_result: VerificationResult,
    *,
    write_uncertain_memory: bool,
) -> bool:
    if verifier_result.label == VerificationLabel.UNCERTAIN:
        return write_uncertain_memory
    return True


def retrieve_selected_memories(
    store: JsonlMemoryStore,
    *,
    query: str,
    top_k_memories: int,
    now: str,
    min_score: float = 0.0,
    min_relevance: float = 0.0,
    max_summary_memories: int | None = None,
) -> list[MemoryRecord]:
    active_records = store.load_active()
    summary_records = store.load_summary()
    ranked = rank_memory_records(active_records + summary_records, query, now=now)
    ranked = sorted(ranked, key=lambda item: item.score, reverse=True)
    selected: list[MemoryRecord] = []
    selected_summary_count = 0
    for item in ranked:
        if not _memory_is_injection_eligible(item.record):
            continue
        if item.score < min_score or item.relevance < min_relevance:
            continue
        if item.record.state == MemoryState.SUMMARY and max_summary_memories is not None:
            if selected_summary_count >= max_summary_memories:
                continue
            selected_summary_count += 1
        selected.append(item.record)
        if len(selected) >= top_k_memories:
            break
    if not selected:
        return []
    selected_ids = {record.memory_id for record in selected}
    updated_active: list[MemoryRecord] = []
    for record in active_records:
        if record.memory_id in selected_ids:
            record = record.model_copy(update={"use_count": record.use_count + 1, "last_used_at": now})
        updated_active.append(record)
    updated_summary: list[MemoryRecord] = []
    for record in summary_records:
        if record.memory_id in selected_ids:
            record = record.model_copy(update={"use_count": record.use_count + 1, "last_used_at": now})
        updated_summary.append(record)
    store.replace_state(MemoryState.ACTIVE, updated_active)
    store.replace_state(MemoryState.SUMMARY, updated_summary)
    return [record for record in updated_active + updated_summary if record.memory_id in selected_ids]


def _memory_is_injection_eligible(record: MemoryRecord) -> bool:
    label = record.verifier_label or record.source_status
    if label == VerificationLabel.UNCERTAIN.value:
        return False
    if record.verifier_score is not None and record.verifier_score < 0.55:
        return False
    return True


def apply_candidate_records(store: JsonlMemoryStore, candidates: list[MemoryRecord]) -> list[dict[str, str]]:
    active_records = store.load_active()
    archive_records = store.load_archive()
    actions: list[dict[str, str]] = []
    provisional_records = store.load_provisional() + candidates
    store.replace_state(MemoryState.PROVISIONAL, provisional_records)

    for candidate in candidates:
        active_records, archived_records, action = resolve_candidate(candidate, active_records)
        archive_records.extend(archived_records)
        actions.append({"memory_id": candidate.memory_id, "action": action})

    kept_provisional = [record for record in provisional_records if record.memory_id not in {c.memory_id for c in candidates}]
    store.replace_state(MemoryState.ACTIVE, active_records)
    store.replace_state(MemoryState.ARCHIVE, archive_records)
    store.replace_state(MemoryState.PROVISIONAL, kept_provisional)
    return actions


def run_periodic_consolidation(store: JsonlMemoryStore) -> dict[str, int]:
    active_before = store.load_active()
    archive_records = store.load_archive()
    active_after, newly_archived = consolidate_active_records(active_before)
    archive_records.extend(newly_archived)
    store.replace_state(MemoryState.ACTIVE, active_after)
    store.replace_state(MemoryState.ARCHIVE, archive_records)
    return {
        "active_before": len(active_before),
        "active_after": len(active_after),
        "archived_delta": len(newly_archived),
    }


def run_budgeted_governance_pass(
    store: JsonlMemoryStore,
    *,
    now: str,
    max_active_records: int,
    similarity_threshold: float,
    cluster_min_size: int,
    retire_failure_days: int,
) -> dict[str, int]:
    active_before = store.load_active()
    summary_before = store.load_summary()
    archive_before = store.load_archive()
    governed = run_budgeted_governance(
        active_records=active_before,
        summary_records=summary_before,
        now=now,
        max_active_records=max_active_records,
        similarity_threshold=similarity_threshold,
        cluster_min_size=cluster_min_size,
        retire_failure_days=retire_failure_days,
    )
    store.replace_state(MemoryState.ACTIVE, governed["active"])
    store.replace_state(MemoryState.SUMMARY, governed["summary"])
    store.replace_state(MemoryState.ARCHIVE, archive_before + governed["archived"])
    return {
        "active_before": len(active_before),
        "active_after": len(governed["active"]),
        "summary_before": len(summary_before),
        "summary_after": len(governed["summary"]),
        "archived_delta": len(governed["archived"]),
    }


def append_memory_health_snapshot(
    store: JsonlMemoryStore,
    *,
    task_id: str,
    event: str,
    active_history: list[int],
) -> dict[str, int | float | str]:
    metrics_dir = store.root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    snapshot = collect_memory_health(
        active_records=store.load_active(),
        summary_records=store.load_summary(),
        provisional_records=store.load_provisional(),
        archive_records=store.load_archive(),
        active_history=active_history,
    )
    payload = {
        "timestamp": _utcnow(),
        "task_id": task_id,
        "event": event,
        **snapshot,
    }
    _append_jsonl(metrics_dir / "memory_health.jsonl", payload)
    return payload


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: str = typer.Option(str(builtin_config_dir / "extra" / "swebench.yaml"), "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    memory_root: str = typer.Option("./memory_v2", "--memory-root", help="Root directory for consolidated memory", rich_help_panel="Memory"),
    top_k_memories: int = typer.Option(2, "--top-k-memories", help="How many active memories to inject", rich_help_panel="Memory"),
    min_memory_score: float = typer.Option(1.2, "--min-memory-score", help="Minimum ranked score before a memory can be injected", rich_help_panel="Memory"),
    min_memory_relevance: float = typer.Option(0.08, "--min-memory-relevance", help="Minimum text relevance before a memory can be injected", rich_help_panel="Memory"),
    max_summary_memories: int = typer.Option(1, "--max-summary-memories", help="Maximum number of summary memories to inject", rich_help_panel="Memory"),
    consolidate_every: int = typer.Option(25, "--consolidate-every", help="Run active-memory consolidation every N completed tasks", rich_help_panel="Memory"),
    max_active_records: int = typer.Option(12, "--max-active-records", help="Hard active-memory budget after governance", rich_help_panel="Memory"),
    summary_cluster_min_size: int = typer.Option(2, "--summary-cluster-min-size", help="Minimum cluster size before summarization", rich_help_panel="Memory"),
    governance_similarity_threshold: float = typer.Option(0.35, "--governance-similarity-threshold", help="Similarity threshold used for summary clustering", rich_help_panel="Memory"),
    retire_failure_days: int = typer.Option(14, "--retire-failure-days", help="Retire unused failed memories older than this many days", rich_help_panel="Memory"),
    disable_memory_write: bool = typer.Option(False, "--disable-memory-write", help="Skip post-run memory induction and writeback", rich_help_panel="Memory"),
    disable_consolidation: bool = typer.Option(False, "--disable-consolidation", help="Disable periodic consolidation", rich_help_panel="Memory"),
    verifier_mode: str = typer.Option("text", "--verifier-mode", help="Verifier mode: off, text, logprob, or auto", rich_help_panel="Verifier"),
    verifier_model: str = typer.Option("gemini-2.5-flash", "--verifier-model", help="Model used by logprob/auto verifier modes", rich_help_panel="Verifier"),
    verifier_reps: int = typer.Option(1, "--verifier-reps", help="Repeated verifications per criterion", rich_help_panel="Verifier"),
    verifier_success_threshold: float = typer.Option(0.70, "--verifier-success-threshold", help="Minimum verifier score for verified_success", rich_help_panel="Verifier"),
    verifier_fail_threshold: float = typer.Option(0.35, "--verifier-fail-threshold", help="Maximum verifier score for verified_fail", rich_help_panel="Verifier"),
    write_uncertain_memory: bool = typer.Option(False, "--write-uncertain-memory", help="Allow uncertain verifier results to write memory", rich_help_panel="Verifier"),
) -> None:
    # fmt: on
    baseline_swebench = _load_baseline_runner()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = baseline_swebench.DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(baseline_swebench.load_dataset(dataset_path, split=split))
    instances = baseline_swebench.filter_instances(
        instances,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        shuffle=shuffle,
    )
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config = yaml.safe_load(get_config_path(config_spec).read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = baseline_swebench.RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")
    memory_lock = threading.Lock()
    run_state = {"completed": 0, "active_history": [0]}

    def process_instance_cm(instance: dict) -> None:
        instance_id = instance["instance_id"]
        instance_dir = output_path / instance_id
        baseline_swebench.remove_from_preds_file(output_path / "preds.json", instance_id)
        (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

        model_obj = baseline_swebench.get_model(config=config.get("model", {}))
        task = instance["problem_statement"]
        store = JsonlMemoryStore(Path(memory_root) / _safe_path_component(model_obj.config.model_name))

        with memory_lock:
            selected_records = retrieve_selected_memories(
                store,
                query=task,
                top_k_memories=top_k_memories,
                now=_utcnow(),
                min_score=min_memory_score,
                min_relevance=min_memory_relevance,
                max_summary_memories=max_summary_memories,
            )
            _append_jsonl(
                store.root / "logs" / "retrieval_events.jsonl",
                {
                    "timestamp": _utcnow(),
                    "task_id": instance_id,
                    "query": task,
                    "selected_memory_ids": [record.memory_id for record in selected_records],
                    "selected_memory_types": [record.memory_type.value for record in selected_records],
                    "selected_count": len(selected_records),
                },
            )
        selected_memory = build_memory_block(selected_records)

        progress_manager.on_instance_start(instance_id)
        progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

        agent = None
        extra_info = None
        exit_status = "RuntimeError"
        result = ""

        try:
            env = baseline_swebench.get_sb_environment(config, instance)
            agent = baseline_swebench.ProgressTrackingAgent(
                model_obj,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )
            exit_status, result = agent.run(task, selected_memory=selected_memory)
        except Exception as exc:
            logger.error(f"Error processing instance {instance_id}: {exc}", exc_info=True)
            exit_status, result = type(exc).__name__, str(exc)
            extra_info = {"traceback": traceback.format_exc()}
        finally:
            baseline_swebench.save_traj(
                agent,
                instance_dir / f"{instance_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=instance_id,
                print_fct=logger.info,
            )
            baseline_swebench.update_preds_file(output_path / "preds.json", instance_id, model_obj.config.model_name, result)
            progress_manager.on_instance_end(instance_id, exit_status)

            if disable_memory_write:
                return

            try:
                messages = agent.messages if agent is not None else json.loads(
                    (instance_dir / f"{instance_id}.traj.json").read_text(encoding="utf-8")
                )["messages"]
                trajectory = _extract_trajectory_text(messages)
                verifier_result = _verify_trajectory_with_model(
                    task=task,
                    trajectory=trajectory,
                    patch=result,
                    exit_status=exit_status,
                    model_obj=model_obj,
                    verifier_mode=verifier_mode,
                    verifier_model=verifier_model,
                    verifier_reps=verifier_reps,
                    verifier_success_threshold=verifier_success_threshold,
                    verifier_fail_threshold=verifier_fail_threshold,
                )
                if not _should_write_memory_for_verification(
                    verifier_result,
                    write_uncertain_memory=write_uncertain_memory,
                ):
                    _append_jsonl(
                        store.root / "logs" / "edit_events.jsonl",
                        {
                            "timestamp": _utcnow(),
                            "task_id": instance_id,
                            "event": "memory_write_abstained",
                            "verifier_label": verifier_result.label.value,
                            "verifier_score": verifier_result.score,
                            "verifier_confidence": verifier_result.confidence,
                            "verifier_criteria": verifier_result.criteria_scores,
                        },
                    )
                    return
                success = verifier_result.label == VerificationLabel.VERIFIED_SUCCESS
                raw_memory_text = _generate_memory_text(
                    task,
                    trajectory,
                    model_obj,
                    success,
                )
                candidates = build_candidate_records(
                    task_id=instance_id,
                    query=task,
                    raw_memory_text=raw_memory_text,
                    source_status="success" if success else "fail",
                    verifier_result=verifier_result,
                )
                with memory_lock:
                    actions = apply_candidate_records(store, candidates)
                    _append_jsonl(
                        store.root / "logs" / "edit_events.jsonl",
                        {
                            "timestamp": _utcnow(),
                            "task_id": instance_id,
                            "source_status": verifier_result.label.value,
                            "verifier_score": verifier_result.score,
                            "verifier_confidence": verifier_result.confidence,
                            "verifier_criteria": verifier_result.criteria_scores,
                            "verifier_model": verifier_result.model_name,
                            "verifier_used_logprobs": verifier_result.used_logprobs,
                            "candidate_count": len(candidates),
                            "actions": actions,
                        },
                    )
                    run_state["completed"] += 1
                    if not disable_consolidation and consolidate_every > 0 and run_state["completed"] % consolidate_every == 0:
                        consolidation_summary = run_periodic_consolidation(store)
                        logger.info(f"[CM] consolidation summary for {instance_id}: {consolidation_summary}")
                    governance_summary = run_budgeted_governance_pass(
                        store,
                        now=_utcnow(),
                        max_active_records=max_active_records,
                        similarity_threshold=governance_similarity_threshold,
                        cluster_min_size=summary_cluster_min_size,
                        retire_failure_days=retire_failure_days,
                    )
                    logger.info(f"[CM] governance summary for {instance_id}: {governance_summary}")
                    active_count = len(store.load_active())
                    run_state["active_history"].append(active_count)
                    snapshot = append_memory_health_snapshot(
                        store,
                        task_id=instance_id,
                        event="post_instance",
                        active_history=run_state["active_history"],
                    )
                logger.info(f"[CM] memory updates for {instance_id}: {actions}")
                logger.info(f"[CM] memory health after {instance_id}: {snapshot}")
            except Exception as exc:
                logger.error(f"[CM] memory writeback failed for {instance_id}: {exc}", exc_info=True)

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as exc:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {exc}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, exc)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance_cm, instance): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
