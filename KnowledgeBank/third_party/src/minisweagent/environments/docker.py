import logging
import os
import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    start_retries: int = int(os.getenv("MSWEA_DOCKER_START_RETRIES", "3"))
    """Number of attempts for transient container startup failures."""
    start_retry_backoff_seconds: float = float(os.getenv("MSWEA_DOCKER_START_RETRY_BACKOFF_SECONDS", "2"))
    """Initial backoff between startup retries."""
    start_retry_max_backoff_seconds: float = float(os.getenv("MSWEA_DOCKER_START_RETRY_MAX_BACKOFF_SECONDS", "8"))
    """Maximum backoff between startup retries."""


class DockerEnvironment:
    def __init__(self, *, config_class: type = DockerEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _cleanup_stale_container(self, container_name: str) -> None:
        subprocess.run(
            [self.config.executable, "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=min(30, self.config.pull_timeout),
            check=False,
        )

    def _format_start_exception(self, exc: BaseException) -> str:
        if isinstance(exc, subprocess.TimeoutExpired):
            return f"timeout after {exc.timeout}s"
        if isinstance(exc, subprocess.CalledProcessError):
            details = " ".join(
                part.strip()
                for part in (exc.stdout or "", exc.stderr or "", str(exc))
                if part and part.strip()
            )
            return details or f"exit code {exc.returncode}"
        return str(exc)

    def _is_retryable_start_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, subprocess.TimeoutExpired):
            return True
        if not isinstance(exc, subprocess.CalledProcessError):
            return False
        if exc.returncode in {125, 126}:
            return True
        details = self._format_start_exception(exc).lower()
        retryable_markers = (
            "timed out",
            "name is already in use",
            "cannot connect to the docker daemon",
            "permission denied while trying to connect",
            "context deadline exceeded",
            "connection reset by peer",
            "temporary failure",
        )
        return any(marker in details for marker in retryable_markers)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        attempts = max(1, self.config.start_retries)
        delay = max(0.0, self.config.start_retry_backoff_seconds)
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            self._cleanup_stale_container(container_name)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.pull_timeout,  # docker pull might take a while
                    check=True,
                )
                self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
                self.container_id = result.stdout.strip()
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_exc = exc
                details = self._format_start_exception(exc)
                if attempt < attempts and self._is_retryable_start_exception(exc):
                    self.logger.warning(
                        "Starting container %s failed on attempt %s/%s; retrying in %.1fs. Details: %s",
                        container_name,
                        attempt,
                        attempts,
                        delay,
                        details,
                    )
                    if delay > 0:
                        time.sleep(delay)
                        delay = min(
                            max(delay * 2, self.config.start_retry_backoff_seconds),
                            self.config.start_retry_max_backoff_seconds,
                        )
                    continue
                self.logger.error(
                    "Starting container %s failed on attempt %s/%s. Details: %s",
                    container_name,
                    attempt,
                    attempts,
                    details,
                )
                raise
        if last_exc is not None:
            raise last_exc

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
