#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-cafe-assistant}"
IMAGE="${IMAGE:?set IMAGE, e.g. ghcr.io/org/cafe-assistant:2026-06-18}"
CANARY_REPLICAS="${CANARY_REPLICAS:-1}"
TENANT_ID="${TENANT_ID:-1}"
CANARY_URL="${CANARY_URL:-}"
RUN_STRICT_EVALS="${RUN_STRICT_EVALS:-0}"

kubectl -n "$NAMESPACE" get deployment/cafe-assistant-api-canary >/dev/null
kubectl -n "$NAMESPACE" get service/cafe-assistant-api-canary >/dev/null

kubectl -n "$NAMESPACE" set image deployment/cafe-assistant-api-canary "api=$IMAGE"
kubectl -n "$NAMESPACE" scale deployment/cafe-assistant-api-canary --replicas="$CANARY_REPLICAS"
kubectl -n "$NAMESPACE" rollout status deployment/cafe-assistant-api-canary --timeout=180s

if [[ -n "$CANARY_URL" ]]; then
  curl -fsS "${CANARY_URL%/}/health" >/dev/null
  TARGET_URL="$CANARY_URL" TENANT_ID="$TENANT_ID" uv run python deploy/shadow_traffic.py
else
  echo "Canary is ready on service cafe-assistant-api-canary."
  echo "Set CANARY_URL and run:"
  echo "  TARGET_URL=http://<canary-url> TENANT_ID=$TENANT_ID uv run python deploy/shadow_traffic.py"
fi

if [[ "$RUN_STRICT_EVALS" == "1" ]]; then
  uv run python evals/run_evals.py --strict
fi

echo "Canary rollout complete for $IMAGE. Promote when ready:"
echo "  IMAGE=$IMAGE CANARY_URL=<canary-url> deploy/promote.sh"
echo "Rollback anytime:"
echo "  deploy/rollback.sh"
