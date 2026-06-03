"""Terminal-Bench oriented runtime for the original KnowledgeBank agent."""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass

from minisweagent.agents.default import DefaultAgent, Submitted


@dataclass
class TerminalBenchRuntimeConfig:
    max_observation_chars: int = 12000
    max_verification_steps: int = 2
    max_repeated_artifact_writes: int = 2
    submit_hint_after_artifact_write: bool = True
    network_recovery_hints: bool = True


_NETWORK_PATTERNS = (
    "connection reset by peer",
    "temporary failure in name resolution",
    "readtimeout",
    "could not resolve",
    "npm err! network",
    "network is unreachable",
    "timed out",
)

_ARTIFACT_WRITE_PATTERNS = (
    r">\s*/app/[^\s;&|]+",
    r">>\s*/app/[^\s;&|]+",
    r"tee\s+(/app/[^\s;&|]+)",
    r"open\(\s*['\"]/app/[^'\"]+['\"]\s*,\s*['\"]w",
    r"write_text\(",
)

_VERIFICATION_PREFIXES = (
    "cat ",
    "ls ",
    "test ",
    "[ ",
    "python ",
    "python3 ",
    "pytest",
    "grep ",
    "rg ",
    "jq ",
    "head ",
    "tail ",
    "file ",
)


def is_artifact_write_command(command: str) -> bool:
    lowered = command.lower()
    if "/app/" not in lowered:
        return False
    return any(re.search(pattern, command) for pattern in _ARTIFACT_WRITE_PATTERNS)


def is_verification_command(command: str) -> bool:
    stripped = command.strip().lower()
    if is_artifact_write_command(command):
        return False
    return stripped.startswith(_VERIFICATION_PREFIXES) or " pytest" in stripped or " python -m pytest" in stripped


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max(1, max_chars // 2)
    tail_chars = max(1, max_chars - head_chars)
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f"\n[terminalbench output truncated: {omitted} chars omitted]\n"
        + text[-tail_chars:]
    )


def summarize_observation(output: dict, *, max_chars: int, network_recovery_hints: bool = True) -> dict:
    summarized = dict(output)
    text = str(summarized.get("output", ""))
    summarized["output"] = _truncate_text(text, max_chars)
    if network_recovery_hints and any(pattern in text.lower() for pattern in _NETWORK_PATTERNS):
        summarized["terminalbench_hint"] = (
            "This looks like an infrastructure/network failure. Prefer configured mirrors, "
            "cached packages, or local files instead of relying on external downloads."
        )
    return summarized


class TerminalBenchRuntimeAgent(DefaultAgent):
    """DefaultAgent runtime with Terminal-Bench feedback shaping.

    This class keeps the original KnowledgeBank model/environment loop intact:
    model.query(), parse_action(), env.execute(), and selected_memory injection
    still come from DefaultAgent. The changes are limited to runtime feedback.
    """

    def __init__(self, *args, terminalbench_config: TerminalBenchRuntimeConfig | dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if terminalbench_config is None:
            terminalbench_config = TerminalBenchRuntimeConfig()
        elif isinstance(terminalbench_config, dict):
            terminalbench_config = TerminalBenchRuntimeConfig(**terminalbench_config)
        self.terminalbench_config = terminalbench_config
        self.artifact_write_seen = False
        self.verification_count = 0
        self.repeated_verification_count = 0
        self.last_verification_command = ""
        self.last_artifact_write_command = ""
        self.repeated_artifact_write_count = 0
        self._ensure_existing_cwd()

    def _ensure_existing_cwd(self) -> None:
        env_config = getattr(self.env, "config", None)
        configured_cwd = getattr(env_config, "cwd", "")
        if configured_cwd and not Path(configured_cwd).exists():
            env_config.cwd = str(Path.cwd())

    def get_observation(self, response: dict) -> dict:
        action = self.parse_action(response)
        command = action["action"]
        if is_artifact_write_command(command):
            self.artifact_write_seen = True
            if command == self.last_artifact_write_command:
                self.repeated_artifact_write_count += 1
            else:
                self.repeated_artifact_write_count = 1
            self.last_artifact_write_command = command
            if self.repeated_artifact_write_count >= self.terminalbench_config.max_repeated_artifact_writes:
                raise Submitted(
                    "TerminalBenchRuntime submitted because the agent repeated the same artifact write "
                    f"{self.repeated_artifact_write_count} times."
                )
        if is_verification_command(command):
            self.verification_count += 1
            if command == self.last_verification_command:
                self.repeated_verification_count += 1
            else:
                self.repeated_verification_count = 1
            self.last_verification_command = command

        output = self.execute_action(action)
        summarized = summarize_observation(
            output,
            max_chars=self.terminalbench_config.max_observation_chars,
            network_recovery_hints=self.terminalbench_config.network_recovery_hints,
        )
        observation = self.render_template(self.config.action_observation_template, output=summarized)
        hint = self._runtime_hint(command, summarized)
        if hint:
            observation = observation.rstrip() + "\n\n<terminalbench_runtime_hint>\n" + hint + "\n</terminalbench_runtime_hint>"
        self.add_message("user", observation)
        return summarized

    def has_finished(self, output: dict[str, str]):
        text = output.get("output", "")
        first_line = text.lstrip().splitlines()[0].strip() if text.strip() else ""
        if first_line == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            raise Submitted("".join(text.lstrip().splitlines(keepends=True)[1:]))
        return super().has_finished(output)

    def _runtime_hint(self, command: str, output: dict) -> str:
        hints: list[str] = []
        if output.get("terminalbench_hint"):
            hints.append(output["terminalbench_hint"])
        if (
            self.terminalbench_config.submit_hint_after_artifact_write
            and self.artifact_write_seen
            and output.get("returncode") == 0
            and is_verification_command(command)
        ):
            hints.append("The requested artifact appears to exist and verification succeeded. Submit now.")
        if self.verification_count > self.terminalbench_config.max_verification_steps:
            hints.append("Do not run another verification command. Submit if the requested artifact/state exists.")
        return "\n".join(dict.fromkeys(hints))
