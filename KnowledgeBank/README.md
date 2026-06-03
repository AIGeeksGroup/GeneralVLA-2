# KnowledgeBank

KnowledgeBank is a research codebase for memory-augmented software agents. It
combines memory consolidation, precision-oriented retrieval, and verifier-guided
judging components on top of a mini-swe-agent style runtime.

This repository contains source code and reproducibility scripts only. Benchmark
datasets, private experiment outputs, run logs, and trajectory files are
intentionally excluded.

## Repository Layout

- `third_party/src/minisweagent`: agent runtime, memory modules, verifier code,
  SWE-Bench adapters, and Terminal-Bench adapter utilities.
- `third_party/tests`: tests for the mini-swe-agent based implementation.
- `SWE-Bench`: scripts for SWE-Bench style runs and report utilities.
- `WebArena`: web-agent pipeline code.

Documentation figures and generated artifacts are not included in this
code-only release.

## Setup

Install the mini-swe-agent based package from source:

```bash
cd third_party
pip install -e .
```

Configure your model provider with environment variables. For OpenAI-compatible
providers:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"
```

Do not commit real API keys, benchmark outputs, trajectories, or local run
directories.

## SWE-Bench

The `SWE-Bench` directory contains runner and reporting scripts. The scripts
expect benchmark data and model credentials to be provided locally by the user.
Generated outputs should be written outside the repository or into ignored
result directories.

## WebArena

The `WebArena` directory contains the web-agent pipeline. Follow the upstream
WebArena and BrowserGym setup instructions for browser and service
configuration before running these scripts.

## Terminal-Bench Adapter

The Terminal-Bench adapter code lives under:

```text
third_party/src/minisweagent/run/extra/
```

The adapter expects Terminal-Bench tasks to be supplied separately. This
repository does not include Terminal-Bench task data or previous experiment
results.

## Development Checks

Useful local checks:

```bash
cd third_party
pytest tests -q
pytest src/minisweagent/run/extra -q
```

Some integration tests require Docker, benchmark datasets, or model API access.

## Acknowledgements

KnowledgeBank builds on ideas and code patterns from:

- Agent-workflow-memory
- WebArena
- mini-swe-agent
- KnowledgeBank-style reasoning memory systems

## Disclaimer

This is a research prototype. It is not intended for production use.
