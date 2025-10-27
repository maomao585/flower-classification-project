# Deployment

This project uses GitHub Actions for CI and image build.

## Branches
- dev, staging, main

## Pipelines
- .github/workflows/python-app.yml: lint + tests on PR
- .github/workflows/docker-image.yml: build image on push/PR; creates .env from secrets on staging/main

## Secrets
- STAGING_ENV: full contents of the .env for staging
- PROD_ENV: full contents of the .env for production

## Branch Protection (manual)
Enable branch protection in GitHub for dev/staging/main so PRs require successful checks.

## Run Locally
```
docker build -t flower-classification:latest .
docker run --rm -p 8000:8000 --env-file .env flower-classification:latest
```
