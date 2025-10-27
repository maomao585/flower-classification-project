# Flower Classification Project

An ML-powered flower image classification app with DevOps and MLOps practices.

- App: FastAPI service exposing /health and /predict
- ML: PyTorch CNN with MLflow tracking
- DevOps: Docker + GitHub Actions CI
- MLOps: MLflow experiments, DVC (to be initialized) for data

## Getting Started

1) Install
```
pip install -r requirements.txt
```
2) Train
```
python train.py
```
3) Run API
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## CI/CD
- On PR to main/dev/staging: lint, format check, tests
- Docker workflow builds image and uses environment secrets to generate .env for staging/main

## MLOps
- MLflow logs under ./mlruns
- DVC: initialize and configure Azure remote, see ml/README_data.md

## Repo Map
- app/: FastAPI app
- src/: data, model, utils
- tests/: unit tests (data, model, api)
- ml/: docs for data & experiments, registry
