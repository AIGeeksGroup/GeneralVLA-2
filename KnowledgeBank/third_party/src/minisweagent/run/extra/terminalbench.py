"""Terminal-Bench adapter for KnowledgeBank's mini-swe-agent path."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import typer
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.default import AgentConfig
from minisweagent.config import get_config_path
from minisweagent.models.litellm_model import LitellmModel

app = typer.Typer(
    help="Run KnowledgeBank through Terminal-Bench's harness.",
    rich_markup_mode="rich",
    add_completion=False,
)


@dataclass
class TmuxSessionEnvironmentConfig:
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 60


class TmuxSessionEnvironment:
    """mini-swe-agent environment backed by a Terminal-Bench tmux session."""

    def __init__(
        self,
        *,
        session,
        config_class: type = TmuxSessionEnvironmentConfig,
        timeout: int | None = None,
        **kwargs,
    ):
        if timeout is not None:
            kwargs["timeout"] = timeout
        self.config = config_class(**kwargs)
        self.session = session
        self._last_pane = ""

    def _wrap_command(self, command: str, cwd: str = "") -> str:
        prefix_parts: list[str] = []
        workdir = cwd or self.config.cwd
        if workdir:
            prefix_parts.append(f"cd {workdir}")
        for key, value in self.config.env.items():
            prefix_parts.append(f"export {key}={json.dumps(str(value))}")
        if not prefix_parts:
            return command
        return " && ".join(prefix_parts + [command])

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        wrapped = self._wrap_command(command, cwd=cwd)
        self.session.send_keys(
            [wrapped, "Enter"],
            block=True,
            max_timeout_sec=self.config.timeout,
        )
        pane = self.session.capture_pane(capture_entire=True)
        self._last_pane = pane
        return {"output": pane, "returncode": 0}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | platform.uname()._asdict() | os.environ


def _load_config(config_path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(get_config_path(config_path).read_text())


def _filter_dataclass_kwargs(config_class: type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in dataclasses.fields(config_class)}
    return {key: value for key, value in values.items() if key in allowed}


def _default_model_factory(**kwargs):
    return LitellmModel(**kwargs)


def _dataset_kwargs(dataset: str) -> dict[str, Any]:
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return {"dataset_path": dataset_path}
    dataset_name, dataset_version = (
        dataset.split("==", 1) if "==" in dataset else (dataset, "head")
    )
    return {"dataset_name": dataset_name, "dataset_version": dataset_version}


try:
    from terminal_bench.agents.base_agent import BaseAgent
except Exception:  # pragma: no cover - lets module import without optional t-bench deps.
    BaseAgent = object


class KnowledgeBankTerminalBenchAgent(BaseAgent):
    """Terminal-Bench custom agent that runs KnowledgeBank's DefaultAgent."""

    @staticmethod
    def name() -> str:
        return "knowledgebank"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        config_path: str | Path = "mini",
        model_factory: Callable[..., Any] | None = None,
        step_limit: int | None = None,
        cost_limit: float | None = None,
        command_timeout_sec: int | None = None,
        **_kwargs,
    ):
        super().__init__(**_kwargs)
        self.model_name = model_name or os.getenv("MSWEA_MODEL_NAME")
        if not self.model_name:
            raise ValueError("model_name is required or MSWEA_MODEL_NAME must be set")
        self.config_path = config_path
        self.model_factory = model_factory or _default_model_factory
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.command_timeout_sec = command_timeout_sec

    def _build_agent(self, session) -> DefaultAgent:
        config = _load_config(self.config_path)
        agent_config = _filter_dataclass_kwargs(AgentConfig, dict(config.get("agent", {})))
        env_config = _filter_dataclass_kwargs(
            TmuxSessionEnvironmentConfig,
            dict(config.get("environment", {})),
        )
        model_config = dict(config.get("model", {}))
        model_config["model_name"] = self.model_name
        if self.step_limit is not None:
            agent_config["step_limit"] = self.step_limit
        if self.cost_limit is not None:
            agent_config["cost_limit"] = self.cost_limit
        if self.command_timeout_sec is not None:
            env_config["timeout"] = self.command_timeout_sec

        model = self.model_factory(**model_config)
        env = TmuxSessionEnvironment(session=session, **env_config)
        return DefaultAgent(model, env, **agent_config)

    def perform_task(self, instruction: str, session, logging_dir: Path | None = None):
        from terminal_bench.agents.base_agent import AgentResult
        from terminal_bench.agents.failure_mode import FailureMode

        agent = self._build_agent(session)
        status, message = agent.run(instruction)
        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "agent": self.name(),
                "model_name": self.model_name,
                "status": status,
                "message": message,
                "messages": agent.messages,
                "model_calls": getattr(agent.model, "n_calls", 0),
                "model_cost": getattr(agent.model, "cost", 0.0),
            }
            (logging_dir / "knowledgebank_trajectory.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return AgentResult(
            total_input_tokens=0,
            total_output_tokens=0,
            failure_mode=FailureMode.NONE,
        )


@app.command()
def main(
    dataset: str = typer.Option("terminal-bench-core==0.1.1", "--dataset"),
    n_tasks: int = typer.Option(3, "--n-tasks"),
    output_path: Path = typer.Option(Path("tb_runs"), "-o", "--output"),
    run_id: str = typer.Option("knowledgebank-terminalbench-smoke"),
    model_name: str = typer.Option(
        os.getenv("MSWEA_MODEL_NAME", "openai/qwen3.5-flash"),
        "-m",
        "--model",
    ),
    config_path: str = typer.Option("mini", "--config"),
    n_concurrent: int = typer.Option(1, "--n-concurrent"),
    agent_timeout_sec: float | None = typer.Option(None, "--agent-timeout-sec"),
) -> None:
    from terminal_bench.harness.harness import Harness

    harness = Harness(
        **_dataset_kwargs(dataset),
        output_path=output_path,
        run_id=run_id,
        agent_import_path="minisweagent.run.extra.terminalbench:KnowledgeBankTerminalBenchAgent",
        model_name=model_name,
        agent_kwargs={"config_path": config_path, "model_name": model_name},
        n_tasks=n_tasks,
        n_concurrent_trials=n_concurrent,
        cleanup=False,
        upload_results=False,
        global_agent_timeout_sec=agent_timeout_sec,
    )
    harness.run()


if __name__ == "__main__":
    app()
