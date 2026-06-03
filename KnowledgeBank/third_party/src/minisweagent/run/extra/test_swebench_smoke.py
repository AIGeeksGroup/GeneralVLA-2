import json
from pathlib import Path

from minisweagent.run.extra import swebench
from minisweagent.run.extra.swebench import _google_genai_model_name


def test_google_genai_model_name_strips_vertex_prefix():
    assert _google_genai_model_name("vertex_ai/gemini-2.5-flash") == "gemini-2.5-flash"


def test_google_genai_model_name_keeps_plain_gemini_name():
    assert _google_genai_model_name("gemini-2.5-flash") == "gemini-2.5-flash"


def test_support_model_name_keeps_deepseek_provider():
    assert swebench._support_model_name("deepseek/deepseek-chat") == "deepseek/deepseek-chat"


def test_safe_update_memory_from_trajectory_swallows_llm_errors(tmp_path: Path, monkeypatch):
    traj_path = tmp_path / "traj.json"
    memory_path = tmp_path / "memory.jsonl"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": "Applied the fix and submitted the patch."},
                ]
            }
        ),
        encoding="utf-8",
    )
    memory_path.write_text("", encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("transient llm failure")

    monkeypatch.setattr(swebench, "llm_judge_status", _boom)

    swebench._safe_update_memory_from_trajectory(
        instance_id="astropy__astropy-1",
        task="Fix the regression",
        model_name="vertex_ai/gemini-2.5-flash",
        traj_path=traj_path,
        memory_path=memory_path,
    )

    assert memory_path.read_text(encoding="utf-8") == ""
