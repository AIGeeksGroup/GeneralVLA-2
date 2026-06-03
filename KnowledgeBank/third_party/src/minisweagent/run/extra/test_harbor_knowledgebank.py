import asyncio
import json
from pathlib import Path

from minisweagent.run.extra import harbor_direct_runner
from minisweagent.run.extra.terminalbench_runtime import TerminalBenchRuntimeAgent
from minisweagent.run.extra.harbor_knowledgebank import (
    KnowledgeBankHarborAgent,
    _agent_dependency_install_command,
    _build_agent_env,
    _system_dependency_install_command,
)


class FakeHarborEnvironment:
    pass


class RecordingHarborAgent(KnowledgeBankHarborAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_exec_calls = []

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        self.agent_exec_calls.append(
            {
                "environment": environment,
                "command": command,
                "env": env or {},
                "cwd": cwd,
                "timeout_sec": timeout_sec,
            }
        )


def test_harbor_adapter_uses_direct_runner_module_instead_of_nested_cli(tmp_path: Path):
    agent = RecordingHarborAgent(
        knowledgebank_src=str(tmp_path),
        config_file=str(tmp_path / "terminalbench.yaml"),
        logs_dir=tmp_path / "logs",
        model_name="openai/test-model",
    )
    context = type("Context", (), {})()

    asyncio.run(agent.run("write /app/result.txt", FakeHarborEnvironment(), context))

    assert len(agent.agent_exec_calls) == 1
    command = agent.agent_exec_calls[0]["command"]
    assert "python3 -m minisweagent.run.extra.harbor_direct_runner" in command
    assert "python3 - <<'PY'" not in command
    assert "<<'PY'" not in command
    assert "</dev/null" not in command
    assert "python3 -m minisweagent.run.mini" not in command
    assert "--task=" not in command
    assert agent.agent_exec_calls[0]["timeout_sec"] is None


def test_build_agent_env_includes_terminalbench_mirrors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.invalid/v1")

    env = _build_agent_env("openai/test-model")

    assert env["PYTHONPATH"] == "/tmp/knowledgebank-src"
    assert env["PIP_INDEX_URL"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    assert env["UV_INDEX_URL"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    assert env["NPM_CONFIG_REGISTRY"] == "https://registry.npmmirror.com"
    assert env["OPENAI_API_KEY"] == "test-key"
    assert env["OPENAI_API_BASE"] == "https://example.invalid/v1"


def test_system_dependency_install_command_skips_apt_when_tools_exist():
    command = _system_dependency_install_command()

    assert "command -v python3" in command
    assert "command -v git" in command
    assert "command -v curl" in command
    assert "command -v pip3" in command
    assert "exit 0" in command
    assert "apt-get" in command
    assert "Acquire::ForceIPv4=true" in command


def test_agent_dependency_install_command_detects_break_system_packages_support():
    command = _agent_dependency_install_command()

    assert "python3 -m pip install --help" in command
    assert "grep -q -- '--break-system-packages'" in command
    assert "PIP_BREAK_SYSTEM_PACKAGES" in command
    assert "${PIP_BREAK_SYSTEM_PACKAGES}" in command
    assert "--ignore-installed" in command
    assert "litellm pyyaml requests" in command


def test_direct_runner_save_trajectory_writes_mini_swe_agent_format(tmp_path: Path):
    class FakeModel:
        n_calls = 2
        cost = 0.03
        config = {"model_name": "openai/test"}

    class FakeEnvironment:
        config = {"cwd": "/app"}

    class FakeAgent:
        config = {"step_limit": 30}
        env = FakeEnvironment()
        model = FakeModel()
        messages = [{"role": "user", "content": "task"}]

    path = tmp_path / "agent" / "mini-swe-agent.trajectory.json"

    harbor_direct_runner.save_trajectory(
        FakeAgent(),
        path,
        exit_status="Submitted",
        result="answer",
    )

    payload = json.loads(path.read_text())
    assert payload["trajectory_format"] == "mini-swe-agent-1"
    assert payload["messages"] == [{"role": "user", "content": "task"}]
    assert payload["info"]["exit_status"] == "Submitted"
    assert payload["info"]["submission"] == "answer"
    assert payload["info"]["model_stats"]["api_calls"] == 2
    assert payload["info"]["model_stats"]["instance_cost"] == 0.03


def test_checkpointing_agent_saves_after_successful_step(monkeypatch, tmp_path: Path):
    calls = []

    class ParentStepAgent:
        messages = [{"role": "assistant", "content": "```bash\ntrue\n```"}]
        model = type("FakeModel", (), {"n_calls": 1, "cost": 0.01})()

        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            return {"output": "ok"}

    class AgentUnderTest(harbor_direct_runner.CheckpointingMixin, ParentStepAgent):
        pass

    def record_save(agent, path, *, exit_status, result):
        calls.append((agent, path, exit_status, result))

    monkeypatch.setattr(harbor_direct_runner, "save_trajectory", record_save)

    path = tmp_path / "trajectory.json"
    output = AgentUnderTest(trajectory_path=path).step()

    assert output == {"output": "ok"}
    assert len(calls) == 1
    saved_agent, saved_path, exit_status, result = calls[0]
    assert isinstance(saved_agent, AgentUnderTest)
    assert saved_path == path
    assert exit_status == "Running"
    assert result == ""


def test_harbor_direct_runner_uses_terminalbench_runtime_agent():
    assert issubclass(harbor_direct_runner.CheckpointingTerminalBenchAgent, TerminalBenchRuntimeAgent)
