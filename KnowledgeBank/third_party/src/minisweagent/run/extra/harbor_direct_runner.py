"""Direct in-container runner for the Harbor Terminal-Bench adapter."""

from __future__ import annotations

import os
import traceback
from dataclasses import fields
from pathlib import Path

import yaml

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.extra.terminalbench_runtime import TerminalBenchRuntimeAgent
from minisweagent.run.utils.save import save_traj


def _filter_dataclass_kwargs(config_class, values: dict) -> dict:
    allowed = {field.name for field in fields(config_class)}
    return {key: value for key, value in values.items() if key in allowed}


def save_trajectory(
    agent: DefaultAgent | None,
    path: Path,
    *,
    exit_status: str | None,
    result: str | None,
    extra_info: dict | None = None,
) -> None:
    save_traj(
        agent,
        path,
        print_path=False,
        exit_status=exit_status,
        result=result,
        extra_info=extra_info,
    )


class CheckpointingMixin:
    def __init__(self, *args, trajectory_path: Path, **kwargs):
        self._trajectory_path = trajectory_path
        super().__init__(*args, **kwargs)

    def step(self) -> dict:
        output = super().step()
        save_trajectory(self, self._trajectory_path, exit_status="Running", result="")
        return output


class CheckpointingTerminalBenchAgent(CheckpointingMixin, TerminalBenchRuntimeAgent):
    pass


class KnowledgeBankDirectRunner:
    def __init__(self, *, instruction: str, model_name: str, config_path: str, trajectory_path: str):
        self.instruction = instruction
        self.model_name = model_name
        self.config_path = config_path
        self.trajectory_path = Path(trajectory_path)

    def run(self) -> None:
        config = yaml.safe_load(get_config_path(self.config_path).read_text())
        agent_config = _filter_dataclass_kwargs(AgentConfig, dict(config.get("agent", {})))
        env_config = dict(config.get("environment", {}))
        model_config = dict(config.get("model", {}))
        terminalbench_config = dict(config.get("terminalbench_runtime", {}))
        model_config["model_name"] = self.model_name

        agent = None
        exit_status = None
        result = None
        extra_info = None
        try:
            model = LitellmModel(**model_config)
            environment = LocalEnvironment(**env_config)
            agent = CheckpointingTerminalBenchAgent(
                model,
                environment,
                trajectory_path=self.trajectory_path,
                terminalbench_config=terminalbench_config,
                **agent_config,
            )
            exit_status, result = agent.run(self.instruction)
        except Exception as exc:
            exit_status = type(exc).__name__
            result = str(exc)
            extra_info = {"traceback": traceback.format_exc()}
            raise
        finally:
            save_trajectory(agent, self.trajectory_path, exit_status=exit_status, result=result, extra_info=extra_info)


def main() -> None:
    KnowledgeBankDirectRunner(
        instruction=os.environ["RB_TASK_INSTRUCTION"],
        model_name=os.environ["RB_MODEL_NAME"],
        config_path=os.environ["RB_CONFIG_PATH"],
        trajectory_path=os.environ["RB_TRAJECTORY_PATH"],
    ).run()


KnowledgeBankDirectRunner = KnowledgeBankDirectRunner


if __name__ == "__main__":
    main()
