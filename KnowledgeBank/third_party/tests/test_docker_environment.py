import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minisweagent.environments.docker import DockerEnvironment


def test_start_container_retries_on_transient_exit_126(monkeypatch):
    run_calls = []
    sleep_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append(cmd)
        if cmd[1] == "rm":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if sum(1 for call in run_calls if len(call) > 1 and call[1] == "run") == 1:
            raise subprocess.CalledProcessError(126, cmd, output="", stderr="transient docker startup failure")
        return subprocess.CompletedProcess(cmd, 0, "container-123\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: None)
    monkeypatch.setattr("minisweagent.environments.docker.time.sleep", lambda seconds: sleep_calls.append(seconds))

    env = DockerEnvironment(image="example:latest", start_retries=3, start_retry_backoff_seconds=0.1)

    assert env.container_id == "container-123"
    assert sum(1 for call in run_calls if len(call) > 1 and call[1] == "run") == 2
    assert sleep_calls == [0.1]


def test_start_container_does_not_retry_non_retryable_error(monkeypatch):
    run_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append(cmd)
        if cmd[1] == "rm":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="unknown image")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: None)
    monkeypatch.setattr("minisweagent.environments.docker.time.sleep", lambda seconds: None)

    try:
        DockerEnvironment(image="missing:latest", start_retries=3, start_retry_backoff_seconds=0.1)
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 1
    else:
        raise AssertionError("Expected non-retryable docker startup failure to be raised")

    assert sum(1 for call in run_calls if len(call) > 1 and call[1] == "run") == 1
