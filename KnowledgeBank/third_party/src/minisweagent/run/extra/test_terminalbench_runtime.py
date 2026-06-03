from pathlib import Path

from minisweagent.run.extra.terminalbench_runtime import (
    TerminalBenchRuntimeAgent,
    TerminalBenchRuntimeConfig,
    is_artifact_write_command,
    is_verification_command,
    summarize_observation,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.n_calls = 0
        self.cost = 0.0

    def query(self, messages):
        self.n_calls += 1
        return {"content": self.responses.pop(0)}

    def get_template_vars(self):
        return {"n_model_calls": self.n_calls, "model_cost": self.cost}


class FakeEnvironment:
    config = {"cwd": "/app"}

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return self.outputs.pop(0)

    def get_template_vars(self):
        return {"cwd": "/app"}


class FakeConfigEnvironment(FakeEnvironment):
    def __init__(self, outputs, cwd):
        super().__init__(outputs)
        self.config = type("Config", (), {"cwd": cwd})()


def test_summarize_observation_truncates_long_output():
    output = {"returncode": 0, "output": "A" * 300 + "\nMIDDLE\n" + "B" * 300}

    summary = summarize_observation(output, max_chars=120)

    assert len(summary["output"]) <= 220
    assert "[terminalbench output truncated" in summary["output"]
    assert "A" * 40 in summary["output"]
    assert "B" * 40 in summary["output"]


def test_summarize_observation_adds_network_hint():
    output = {"returncode": 1, "output": "pip failed: Connection reset by peer"}

    summary = summarize_observation(output, max_chars=500)

    assert "infrastructure/network" in summary["terminalbench_hint"]
    assert "mirrors" in summary["terminalbench_hint"]


def test_command_classifiers_detect_artifact_writes_and_verification():
    assert is_artifact_write_command("echo answer > /app/result.txt")
    assert is_artifact_write_command("python3 - <<'PY'\nopen('/app/out.json','w').write('{}')\nPY")
    assert not is_artifact_write_command("cat /app/result.txt")

    assert is_verification_command("python3 -m pytest -q")
    assert is_verification_command("cat /app/result.txt")
    assert not is_verification_command("echo answer > /app/result.txt")


def test_runtime_hints_submit_after_artifact_write_then_successful_verification():
    model = FakeModel(
        [
            "```bash\necho answer > /app/result.txt\n```",
            "```bash\ncat /app/result.txt\n```",
            "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
        ]
    )
    env = FakeEnvironment(
        [
            {"returncode": 0, "output": ""},
            {"returncode": 0, "output": "answer\n"},
            {"returncode": 0, "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"},
        ]
    )
    agent = TerminalBenchRuntimeAgent(
        model,
        env,
        terminalbench_config=TerminalBenchRuntimeConfig(max_observation_chars=500),
    )

    status, _ = agent.run("create /app/result.txt")

    assert status == "Submitted"
    assert any("Submit now" in message["content"] for message in agent.messages if message["role"] == "user")
    assert env.commands == [
        "echo answer > /app/result.txt",
        "cat /app/result.txt",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    ]


def test_runtime_hints_stop_after_repeated_verification():
    model = FakeModel(
        [
            "```bash\necho answer > /app/result.txt\n```",
            "```bash\ncat /app/result.txt\n```",
            "```bash\ncat /app/result.txt\n```",
            "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
        ]
    )
    env = FakeEnvironment(
        [
            {"returncode": 0, "output": ""},
            {"returncode": 0, "output": "answer\n"},
            {"returncode": 0, "output": "answer\n"},
            {"returncode": 0, "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"},
        ]
    )
    agent = TerminalBenchRuntimeAgent(
        model,
        env,
        terminalbench_config=TerminalBenchRuntimeConfig(max_verification_steps=1),
    )

    status, _ = agent.run("create /app/result.txt")

    assert status == "Submitted"
    assert any("Do not run another verification" in message["content"] for message in agent.messages)


def test_runtime_submits_after_repeated_identical_artifact_writes():
    repeated_write = "```bash\necho answer > /app/result.txt\n```"
    model = FakeModel([repeated_write, repeated_write, repeated_write])
    env = FakeEnvironment(
        [
            {"returncode": 0, "output": ""},
            {"returncode": 0, "output": ""},
            {"returncode": 0, "output": ""},
        ]
    )
    agent = TerminalBenchRuntimeAgent(
        model,
        env,
        terminalbench_config=TerminalBenchRuntimeConfig(max_repeated_artifact_writes=2),
    )

    status, message = agent.run("create /app/result.txt")

    assert status == "Submitted"
    assert "repeated the same artifact write" in message
    assert env.commands == ["echo answer > /app/result.txt"]


def test_runtime_falls_back_when_configured_cwd_is_missing(tmp_path):
    model = FakeModel(
        [
            "```bash\npwd\n```",
            "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
        ]
    )
    env = FakeConfigEnvironment(
        [
            {"returncode": 0, "output": str(tmp_path) + "\n"},
            {"returncode": 0, "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"},
        ],
        cwd=str(tmp_path / "missing-app"),
    )
    agent = TerminalBenchRuntimeAgent(model, env)

    status, _ = agent.run("show cwd")

    assert status == "Submitted"
    assert env.config.cwd == str(Path.cwd())
