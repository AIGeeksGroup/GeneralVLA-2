import importlib
from pathlib import Path

from minisweagent.memory_v2.schema import MemoryState, MemoryType
from minisweagent.memory_v2.store import JsonlMemoryStore
from minisweagent.verifier.schema import VerificationLabel, VerificationResult


def test_swebench_cm_module_imports():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    assert hasattr(mod, "app")


def test_parse_memory_items_and_build_candidates():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    raw = """
# Memory Item 1
## Title Inspect tests first
## Description Start from the failing regression test before changing code.
## Content Reproduce the issue, inspect the exact failing assertion, and only then patch the implementation.
"""
    items = mod.parse_memory_items(raw)
    assert len(items) == 1
    records = mod.build_candidate_records(
        task_id="astropy__astropy-1",
        query="Fix the failing regression",
        raw_memory_text=raw,
        source_status="success",
        created_at="2026-04-01T00:00:00Z",
    )
    assert len(records) == 1
    assert records[0].state == MemoryState.PROVISIONAL
    assert records[0].memory_type in {MemoryType.PROCEDURAL_HINT, MemoryType.TOOL_USAGE}


def test_build_candidate_records_uses_verifier_result_for_quality_and_status():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    raw = """
# Memory Item 1
## Title Verify before editing again
## Description Trust observed command output over narration.
## Content If the agent edits after the last passing command, treat the final patch as unverified.
"""
    verifier_result = VerificationResult(
        label=VerificationLabel.UNCERTAIN,
        score=0.49,
        confidence=0.02,
        criteria_scores={"root_cause": 0.6, "code_review": 0.5, "empirical_verification": 0.37},
        raw_outputs=["<score>K</score>"],
        model_name="fake-verifier",
        n_reps=1,
        used_logprobs=False,
        exit_status="Submitted",
    )

    records = mod.build_candidate_records(
        task_id="astropy__astropy-2",
        query="Fix the failing regression",
        raw_memory_text=raw,
        source_status="success",
        verifier_result=verifier_result,
        created_at="2026-04-01T00:00:00Z",
    )

    assert len(records) == 1
    assert records[0].source_status == "uncertain"
    assert records[0].quality_score == 0.49
    assert records[0].confidence == 0.02
    assert records[0].verifier_score == 0.49
    assert records[0].verifier_label == "uncertain"
    assert records[0].verifier_model == "fake-verifier"


class FakeScoringModel:
    def __init__(self, score_token: str):
        self.config = type("Config", (), {"model_name": "fake-agent-model"})()
        self.score_token = score_token
        self.prompts = []

    def query(self, messages, **_kwargs):
        self.prompts.append(messages[-1]["content"])
        return {"content": f"Reasoning.\n<score>{self.score_token}</score>"}


def test_verify_trajectory_with_model_uses_text_verifier_client():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    result = mod._verify_trajectory_with_model(
        task="Fix a regression",
        trajectory="Ran reproducer and pytest. All passed.",
        patch="diff --git a/file.py b/file.py",
        exit_status="Submitted",
        model_obj=FakeScoringModel("A"),
        verifier_mode="text",
        verifier_model="fake-verifier",
        verifier_reps=1,
    )

    assert result.label == VerificationLabel.VERIFIED_SUCCESS
    assert result.score == 1.0
    assert len(result.criteria_scores) == 3


def test_should_write_memory_for_verification_abstains_on_uncertain_by_default():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    result = VerificationResult(
        label=VerificationLabel.UNCERTAIN,
        score=0.5,
        confidence=0.0,
        criteria_scores={"root_cause": 0.5},
        raw_outputs=[],
        model_name="fake",
        n_reps=1,
        used_logprobs=False,
    )

    assert mod._should_write_memory_for_verification(result, write_uncertain_memory=False) is False
    assert mod._should_write_memory_for_verification(result, write_uncertain_memory=True) is True


def test_retrieve_selected_memories_can_use_summary_records(tmp_path: Path):
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    store = JsonlMemoryStore(tmp_path)
    store.add(
        mod.MemoryRecord(
            memory_id="sum-1",
            task_id="t1",
            query="fix astropy regression",
            content="Summary:\n- Inspect the failing regression test before patching.",
            memory_type=MemoryType.PROCEDURAL_HINT,
            source_status="success",
            state=MemoryState.SUMMARY,
            confidence=0.9,
            quality_score=0.9,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="summary:inspect-regression",
        )
    )
    selected = mod.retrieve_selected_memories(
        store,
        query="fix astropy regression",
        top_k_memories=1,
        now="2026-04-03T00:00:00Z",
    )
    assert len(selected) == 1
    assert selected[0].state == MemoryState.SUMMARY


def test_retrieve_selected_memories_can_filter_low_relevance_memories(tmp_path: Path):
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    store = JsonlMemoryStore(tmp_path)
    store.add(
        mod.MemoryRecord(
            memory_id="sum-1",
            task_id="t1",
            query="debug shell tooling",
            content="Summary:\n- Use shell scripts to batch-process logs.",
            memory_type=MemoryType.TOOL_USAGE,
            source_status="success",
            state=MemoryState.SUMMARY,
            confidence=0.9,
            quality_score=0.9,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="summary:shell-logs",
        )
    )
    selected = mod.retrieve_selected_memories(
        store,
        query="fix astropy regression in table validation",
        top_k_memories=2,
        now="2026-04-03T00:00:00Z",
        min_score=1.2,
        min_relevance=0.08,
        max_summary_memories=1,
    )
    assert selected == []


def test_retrieve_selected_memories_filters_low_verifier_score_even_when_relevant(tmp_path: Path):
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    store = JsonlMemoryStore(tmp_path)
    store.add(
        mod.MemoryRecord(
            memory_id="low-verifier",
            task_id="t1",
            query="fix astropy regression in table validation",
            content="Inspect astropy table validation and patch the failing regression.",
            memory_type=MemoryType.PROCEDURAL_HINT,
            source_status="uncertain",
            state=MemoryState.ACTIVE,
            confidence=0.9,
            quality_score=0.9,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="procedural_hint:astropy-table-validation",
            verifier_score=0.49,
            verifier_label="uncertain",
        )
    )
    selected = mod.retrieve_selected_memories(
        store,
        query="fix astropy regression in table validation",
        top_k_memories=1,
        now="2026-04-03T00:00:00Z",
        min_score=0.0,
        min_relevance=0.0,
    )

    assert selected == []


def test_retrieve_selected_memories_caps_summary_count(tmp_path: Path):
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    store = JsonlMemoryStore(tmp_path)
    for idx in range(2):
        store.add(
            mod.MemoryRecord(
                memory_id=f"sum-{idx}",
                task_id=f"t{idx}",
                query="fix astropy regression",
                content="Summary:\n- Inspect the failing regression test before patching.",
                memory_type=MemoryType.PROCEDURAL_HINT,
                source_status="success",
                state=MemoryState.SUMMARY,
                confidence=0.9,
                quality_score=0.9,
                created_at="2026-04-01T00:00:00Z",
                dedup_key=f"summary:inspect-regression-{idx}",
            )
        )
    selected = mod.retrieve_selected_memories(
        store,
        query="fix astropy regression",
        top_k_memories=3,
        now="2026-04-03T00:00:00Z",
        min_score=0.0,
        min_relevance=0.0,
        max_summary_memories=1,
    )
    assert len(selected) == 1
    assert selected[0].state == MemoryState.SUMMARY


def test_build_candidate_records_normalizes_semantic_dedup_keys():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    raw_a = """
# Memory Item 1
## Title Choose the Right Tool for Code Modification
## Description For complex Python edits, programmatic tools are more reliable than shell utilities.
## Content Shell commands like `sed` are brittle for indentation-sensitive multi-line edits. Prefer a Python script.
"""
    raw_b = """
# Memory Item 1
## Title Robust Multi-Line Code Modification
## Description For complex multi-line Python edits, use a dedicated Python script to replace whole blocks.
## Content Text tools like `sed` are error-prone for indentation-sensitive changes.
"""
    rec_a = mod.build_candidate_records(
        task_id="astropy__astropy-a",
        query="fix astropy transform bug",
        raw_memory_text=raw_a,
        source_status="success",
        created_at="2026-04-01T00:00:00Z",
    )[0]
    rec_b = mod.build_candidate_records(
        task_id="astropy__astropy-b",
        query="fix astropy table bug",
        raw_memory_text=raw_b,
        source_status="success",
        created_at="2026-04-01T00:00:00Z",
    )[0]
    assert rec_a.dedup_key == rec_b.dedup_key


def test_support_model_queries_use_deterministic_parameters():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    calls = []

    class FakeModel:
        def query(self, messages, **kwargs):
            calls.append({"messages": messages, "kwargs": kwargs})
            return {"content": "success"}

    model = FakeModel()
    mod._judge_status_with_model("task", "trajectory", model)
    mod._generate_memory_text("task", "trajectory", model, True)

    assert len(calls) == 2
    assert calls[0]["kwargs"]["temperature"] == 0.0
    assert calls[0]["kwargs"]["max_tokens"] <= 32
    assert calls[1]["kwargs"]["temperature"] == 0.0
    assert calls[1]["kwargs"]["max_tokens"] >= 256


def test_extract_trajectory_text_skips_submission_marker():
    mod = importlib.import_module("minisweagent.run.extra.swebench_cm")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "<pr_description>task</pr_description>"},
        {"role": "assistant", "content": "THOUGHT: inspect\n```bash\nrg -n \"foo\" .\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>ok</output>"},
        {
            "role": "assistant",
            "content": "THOUGHT: submit\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
        },
    ]

    trajectory = mod._extract_trajectory_text(messages)

    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in trajectory
    assert "rg -n" in trajectory
