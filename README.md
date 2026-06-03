# GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning


This repository contains three code components used in the GeneralVLA project:

- `GeneralVLA/`: robot memory VLA runtime, model configuration, evaluation tools, and tests.
- `GeoFuse-MV3D/`: multi-view 3D geometry fusion and evaluation utilities.
- `KnowledgeBank/`: memory-augmented software-agent code and benchmark scripts.

Large checkpoints, datasets, generated outputs, robot logs, and benchmark
trajectories are intentionally not included. Each component documents its own
setup steps and expected external assets.

## Repository Layout

```text
.
├── GeneralVLA/
├── GeoFuse-MV3D/
├── KnowledgeBank/
├── THIRD_PARTY_NOTICES.md
└── README.md
```

## Quick Start

Install and run each component from its own directory:

```bash
cd GeneralVLA
bash scripts/bootstrap.sh
pytest -q
```

```bash
cd GeoFuse-MV3D
pip install -r requirements.txt
python scripts/run_full_pipeline.py --config configs/paths.local.yaml
```

```bash
cd KnowledgeBank/third_party
pip install -e .
pytest tests -q
```

See the README inside each subdirectory for detailed asset paths, model
configuration, and benchmark-specific instructions.

## External Assets

The repository is code-only. Before running the full pipelines, prepare the
external assets described by each component, including model checkpoints,
benchmark datasets, WebArena services, GSO assets, and robot/runtime-specific
configuration.

Component asset entry points:

- `GeneralVLA/`: project model assets are expected from
  `https://huggingface.co/AIGeeksGroup/GeneralVLA`.
- `GeoFuse-MV3D/`: use the official upstream assets documented in
  `GeoFuse-MV3D/docs/external_assets.md`.
- `KnowledgeBank/`: use the official benchmark/model-provider assets documented
  in `KnowledgeBank/README.md`.

Do not commit API keys, model checkpoints, local datasets, generated results, or
private trajectories.
