# KnowledgeBank Architecture

KnowledgeBank is organized around a mini-swe-agent compatible runtime with three
research components:

1. Memory modules that store and reuse reasoning traces from previous agent
   attempts.
2. Retrieval and precision utilities that select relevant memories for the
   current task.
3. Verifier-guided judging utilities that help score or filter candidate
   reasoning and actions.

The repository intentionally excludes benchmark datasets and private experiment
outputs. To reproduce experiments, provide benchmark data locally and write run
artifacts outside this source tree or into ignored output directories.

## Runtime Adapters

SWE-Bench and Terminal-Bench adapters live under
`third_party/src/minisweagent/run/extra`. These adapters connect benchmark
harnesses to the local agent runtime while keeping benchmark data external to the
repository.
