# Workflows

## Run Tests

```bash
pytest
```

## Run a Single Dataset Experiment

```bash
python main.py \
  --mode dataset \
  --log data/Sepsis_cases.xes \
  --model data/Sepsis_cases_model.pkl \
  --algorithms forward backward bidir_std dibbs \
  --heuristics zero \
  --timeout 60 \
  --output-dir outputs/sepsis_zero_smoke
```

Use `--trace-hash-allowlist` to rerun only selected trace variants:

```bash
python main.py \
  --mode dataset \
  --log data/Sepsis_cases.xes \
  --model data/Sepsis_cases_model.pkl \
  --trace-hash-allowlist outputs/allowlists/sepsis_subset.txt \
  --output-dir outputs/sepsis_subset
```

## Aggregate Results

```bash
python scripts/aggregate_astar_results.py \
  --root-dir outputs \
  --output-csv outputs/aggregate_results.csv
```

## Build Feature Tables

```bash
python scripts/feature_engineering.py \
  --driver-csv outputs/driver.csv \
  --aggregate-csv outputs/aggregate_results.csv \
  --out-dir outputs/features
```

## Analyze Method Recommendations

The paper's runtime-oriented recommendation rule uses runtime as the primary
metric:

```bash
python scripts/analyze_setting_recommendations.py \
  --results-root outputs \
  --out-dir outputs/setting_recommendations \
  --primary-metric time_seconds
```

For the expansion-count comparison, switch the primary metric:

```bash
python scripts/analyze_setting_recommendations.py \
  --results-root outputs \
  --out-dir outputs/setting_recommendations_expansions \
  --primary-metric expansions
```
