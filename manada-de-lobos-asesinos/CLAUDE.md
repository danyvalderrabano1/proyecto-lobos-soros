# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Financial market prediction ML project with two parallel model stacks:
1. **DVC pipeline** — sklearn RandomForest + walk-forward validation, orchestrated via `dvc.yaml`
2. **LoboNet** — custom PyTorch MLP for binary classification, living in `src/lobonet/`

## Common commands

### DVC pipeline (primary workflow)
```bash
# Generate synthetic data and run full pipeline
dvc repro

# Run individual stages
python src/main.py synthetic          # generate data/processed/processed.csv
python src/main.py preprocess         # preprocess from data/raw/*.csv instead

python src/walkforward.py             # time-series cross-validation (baseline)
python src/train.py                   # train RandomForest + log to MLflow
python src/predict.py --evaluate      # evaluate model
python src/predict.py --predict       # generate predictions CSV
```

### LoboNet (PyTorch MLP)
```bash
# Run from src/ so imports resolve correctly
cd src
python -m lobonet.train               # train, saves lobonet_model.pth
python -m lobonet.predict             # inference demo (requires lobonet_model.pth)
python -m lobonet.tune                # Optuna hyperparameter search + MLflow logging
```

### MLflow UI
```bash
# For src/train.py runs (sqlite backend, project root)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# For tune.py runs (file backend, src/mlruns/)
mlflow ui --backend-store-uri file:///path/to/src/mlruns
```

## Architecture

### DVC pipeline (`dvc.yaml`)
Three sequential stages:
- `synthetic` → `src/main.py synthetic` → `data/processed/processed.csv`
- `walkforward` → `src/walkforward.py` → `reports/walkforward_metrics.json`, `reports/walkforward_folds.csv`
- `train` → `src/train.py` → `models/model.pkl`, `reports/train_metrics.json`

### `src/lobonet/` package
| File | Purpose |
|------|---------|
| `config.py` | Global constants: `INPUT_SIZE=10`, `EPOCHS=50`, `LEARNING_RATE=0.001` |
| `model.py` | `LoboNet`: 3-layer MLP, sigmoid output for binary classification |
| `dataset.py` | Synthetic data loader returning train/test tensors |
| `train.py` | Standalone training loop, saves `lobonet_model.pth` checkpoint |
| `predict.py` | Loads checkpoint from `CKPT_PATH="lobonet_model.pth"`, runs demo inference |
| `tune.py` | Optuna study (TPE sampler, 30 trials), nested MLflow runs in `src/mlruns/` |

### Standalone scripts (`src/`)
- `main.py` — data ingestion: `preprocess` reads `data/raw/*.csv` (multi-encoding robust), `synthetic` generates a random-walk price series
- `train.py` — sklearn `RandomForestRegressor`, logs to `sqlite:///mlflow.db`, experiment `lobo_ai_experiments`
- `walkforward.py` — time-series splitting with configurable `min_train_size/test_size/step_size`, baseline majority-class predictor, outputs JSON + CSV metrics
- `predict.py` — placeholder evaluate/predict stubs that require `models/lobonet_model.pth`

### MLflow tracking — two separate backends
- `src/train.py` → `sqlite:///mlflow.db` (project root), experiment `lobo_ai_experiments`
- `src/lobonet/tune.py` → `file:///…/src/mlruns`, experiment `LoboNet_Optuna_Tune`

### Data flow
```
data/raw/*.csv
    └─ src/main.py preprocess ──► data/processed/processed.csv
                                        ├─ src/walkforward.py ──► reports/
                                        └─ src/train.py ──► models/model.pkl
```

## Key notes

- The `src/lobonet/` package must be imported as `lobonet.*`; always run from `src/` or ensure `src/` is on `PYTHONPATH`.
- `src/predict.py` (DVC stage) expects `models/lobonet_model.pth`, but `src/lobonet/train.py` saves to `src/lobonet_model.pth` — these paths are not yet connected.
- Walk-forward validation currently uses a naive baseline predictor (majority class / mean); plug in a real model inside `walkforward.py`'s fold loop.
- `src/lobonet/features.py` is empty — feature engineering not yet implemented.
- The `dvc.yaml` `train` stage uses `--model logistic` as default but the script only implements `RandomForestRegressor`.
