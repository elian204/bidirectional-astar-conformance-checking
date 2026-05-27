# Data and Artifacts

This repository contains the code and curated inputs used by the conformance
search-direction study. The raw experiment outputs are intentionally kept out of
Git because full runs can produce many gigabytes of CSV, JSON, and log files.

## Tracked Inputs

The `data/` directory currently tracks a curated subset of event logs and
discovered models needed by the existing scripts and tests. Two tracked `.xes`
files are larger than GitHub's recommended 50 MB file size:

- `data/BPI_Challenge_2012.xes`
- `data/prDm6.xes`

They are already present in repository history. Removing them in a normal commit
would not shrink clone size; that would require a history rewrite and a force
push, or a Git LFS migration.

## Ignored Outputs

The following are treated as generated artifacts and should remain outside Git:

- `results/`
- `tmp_smoke/`
- `outputs/`
- `artifacts/`
- `discovery/models/`
- `experiments/results_*/`
- `experiments/hm_sound_safe_batches_*/`
- `experiments/targeted_hardcase_*/`
- `*.log`, `*.err`, `*.out`

If a result table is needed for a paper revision, prefer committing a small
curated CSV or Markdown summary rather than a full run directory.

## Reproducibility Notes

Use `python main.py --help` for the experiment runner. The analysis scripts in
`scripts/` expect aggregate CSVs produced by the runner or by
`scripts/aggregate_astar_results.py`.

The marking-equation heuristic depends on Gurobi. Runs that only use the zero
heuristic do not require an active Gurobi license.

## Future Cleanup Option

A Git LFS migration would make the repository lighter for future clones, but it
rewrites history. Do that only when collaborators are ready to re-clone or reset
their local checkouts.
