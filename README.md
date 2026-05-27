# Bidirectional A* for Conformance Checking

Code and experiment utilities for the paper:

> Unidirectional vs Bidirectional A* for Conformance Checking: A Theoretical Analysis and Empirical Study

The repository implements forward, backward, and bidirectional A* variants on
synchronous-product reachability graphs, together with zero, marking-equation,
and MMR/REACH heuristic configurations. It also contains the scripts used to
build experiment tables, extract structural features, and train the shallow
method-selection rules reported in the paper.

## Repository Layout

- `algorithms/`: forward, backward, standard bidirectional, and DIBBS search.
- `core/`: Petri-net, trace-model, and synchronous-product data structures.
- `heuristics/`: zero, marking-equation, and REACH-style heuristic code.
- `experiments/`: experiment runner, model loading, and method dispatch.
- `scripts/`: aggregation, feature engineering, and analysis utilities.
- `discovery/`: process-model discovery and model-selection helper scripts.
- `docs/`: repository, data, and experiment-output notes.
- `tests/`: regression tests for search, feature extraction, and pipelines.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi is required for marking-equation runs and must be installed with a valid
license. Zero-heuristic runs and most tests do not require an active Gurobi
license.

## Basic Usage

Run an experiment with the command-line entry point:

```bash
python main.py --help
```

Run the test suite:

```bash
pytest
```

For data and artifact conventions, see `docs/DATA_AND_ARTIFACTS.md`.

Generated experiment outputs can be large and are intentionally ignored by
default. Keep long-running result directories outside version control unless
they are deliberately curated for release.
