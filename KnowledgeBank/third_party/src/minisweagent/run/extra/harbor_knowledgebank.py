"""Harbor agent adapter for running local KnowledgeBank mini-swe-agent code."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path, PurePosixPath

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from harbor.agents.installed.mini_swe_agent import convert_and_save_trajectory
from harbor.agents.utils import get_api_key_var_names_from_model_name
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


def _build_agent_env(model_name: str) -> dict[str, str]:
    env = {
        "PYTHONPATH": "/tmp/knowledgebank-src",
        "MSWEA_CONFIGURED": "true",
        "MSWEA_COST_TRACKING": "ignore_errors",
        "PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "UV_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "UV_PYTHON_INSTALL_MIRROR": (
            "https://gh-proxy.com/https://github.com/astral-sh/"
            "python-build-standalone/releases/download"
        ),
        "UV_HTTP_TIMEOUT": "120",
        "NPM_CONFIG_REGISTRY": "https://registry.npmmirror.com",
        "DEBIAN_FRONTEND": "noninteractive",
        "APT_OPTS": "-o Acquire::ForceIPv4=true",
    }
    if "MSWEA_API_KEY" in os.environ:
        env["MSWEA_API_KEY"] = os.environ["MSWEA_API_KEY"]
    else:
        for key in get_api_key_var_names_from_model_name(model_name):
            if key in os.environ:
                env[key] = os.environ[key]
    for key in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _system_dependency_install_command() -> str:
    return (
        "set -e; "
        "if command -v python3 >/dev/null 2>&1 && "
        "(command -v pip3 >/dev/null 2>&1 || python3 -m pip --version >/dev/null 2>&1) && "
        "command -v git >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then "
        "exit 0; "
        "fi; "
        "if command -v apt-get >/dev/null 2>&1; then "
        "sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; "
        "s|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' "
        "/etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; "
        "apt-get -o Acquire::ForceIPv4=true update && "
        "apt-get -o Acquire::ForceIPv4=true install -y python3 python3-pip python3-venv git curl; "
        "fi"
    )


def _agent_dependency_install_command() -> str:
    packages = (
        "litellm pyyaml requests rich typer jinja2 tenacity python-dotenv "
        "platformdirs textual prompt_toolkit openai"
    )
    return (
        "set -e; "
        "PIP_BREAK_SYSTEM_PACKAGES=''; "
        "if python3 -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'; then "
        "PIP_BREAK_SYSTEM_PACKAGES='--break-system-packages'; "
        "fi; "
        f"python3 -m pip install ${{PIP_BREAK_SYSTEM_PACKAGES}} --ignore-installed --quiet {packages}"
    )


class KnowledgeBankHarborAgent(BaseInstalledAgent):
    """Harbor custom agent that runs this repo's mini-swe-agent/KnowledgeBank path."""

    SUPPORTS_ATIF: bool = True

    CLI_FLAGS = [
        CliFlag("cost_limit", cli="--cost-limit", type="str", default="0"),
    ]

    def __init__(
        self,
        knowledgebank_src: str | None = None,
        config_file: str | None = None,
        reasoning_effort: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        default_src = Path(__file__).resolve().parents[3]
        self._knowledgebank_src = Path(knowledgebank_src) if knowledgebank_src else default_src
        self._config_file = (
            Path(config_file)
            if config_file
            else default_src / "minisweagent/config/extra/terminalbench.yaml"
        )
        self._reasoning_effort = reasoning_effort

    @staticmethod
    def name() -> str:
        return "knowledgebank"

    def get_version_command(self) -> str | None:
        return "python3 - <<'PY'\nimport minisweagent; print(getattr(minisweagent, '__version__', 'local'))\nPY"

    @property
    def _mini_swe_agent_trajectory_path(self) -> PurePosixPath:
        return EnvironmentPaths.agent_dir / "mini-swe-agent.trajectory.json"

    @property
    def _atif_trajectory_path(self) -> PurePosixPath:
        return EnvironmentPaths.agent_dir / "trajectory.json"

    async def install(self, environment: BaseEnvironment) -> None:
        if not self._knowledgebank_src.exists():
            raise FileNotFoundError(f"Missing knowledgebank_src: {self._knowledgebank_src}")
        if not self._config_file.exists():
            raise FileNotFoundError(f"Missing config_file: {self._config_file}")

        await self.exec_as_root(
            environment,
            command=_system_dependency_install_command(),
            env={"DEBIAN_FRONTEND": "noninteractive"},
            timeout_sec=300,
        )
        await environment.upload_dir(self._knowledgebank_src, "/tmp/knowledgebank-src")
        await environment.upload_file(str(self._config_file), "/tmp/knowledgebank-terminalbench.yaml")

        await self.exec_as_agent(
            environment,
            command=_agent_dependency_install_command(),
            env={"PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple"},
            timeout_sec=900,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        mini_path = self.logs_dir / "mini-swe-agent.trajectory.json"
        if not mini_path.exists():
            return
        try:
            trajectory = json.loads(mini_path.read_text())
        except Exception:
            return

        info = trajectory.get("info") or {}
        stats = info.get("model_stats") or {}
        context.cost_usd = stats.get("instance_cost") or 0
        context.metadata = {"knowledgebank_status": info.get("exit_status")}
        try:
            convert_and_save_trajectory(mini_path, self.logs_dir / "trajectory.json", str(uuid.uuid4()))
        except Exception:
            pass

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in provider/model format")

        env = _build_agent_env(self.model_name)
        env["RB_TASK_INSTRUCTION"] = instruction
        env["RB_MODEL_NAME"] = self.model_name
        env["RB_CONFIG_PATH"] = "/tmp/knowledgebank-terminalbench.yaml"
        env["RB_TRAJECTORY_PATH"] = str(self._mini_swe_agent_trajectory_path)

        if self._reasoning_effort:
            env["RB_REASONING_EFFORT"] = self._reasoning_effort

        await self.exec_as_agent(
            environment,
            command=(
                "python3 -m minisweagent.run.extra.harbor_direct_runner "
                "2>&1 | tee /logs/agent/mini-swe-agent.txt"
            ),
            env=env,
            timeout_sec=None,
        )
