#!/usr/bin/env bash
set -euo pipefail

IMG=${1:-flower-classification:latest}
NAME=${2:-flower-api}
PORT=${3:-8000}

# build (optional)
if [[ "${BUILD:-1}" == "1" ]]; then
  docker build -t "$IMG" .
fi

# cleanup
(docker rm -f "$NAME" >/dev/null 2>&1) || true

# run
docker run -d --name "$NAME" -p ${PORT}:8000 "$IMG"

echo "Waiting for service to be ready..."
for i in {1..45}; do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null; then
    echo "Service is up"
    curl -sf "http://localhost:${PORT}/health"
    exit 0
  fi
  sleep 2
  echo -n "."
done

echo "Healthcheck failed after retries"
docker ps -a || true
docker logs "$NAME" || true
docker rm -f "$NAME" || true
exit 56
