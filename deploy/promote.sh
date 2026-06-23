#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-cafe-assistant}"
IMAGE="${IMAGE:?set IMAGE promoted from canary}"
STABLE_REPLICAS="${STABLE_REPLICAS:-3}"
TENANT_ID="${TENANT_ID:-1}"
CANARY_URL="${CANARY_URL:-}"

kubectl -n "$NAMESPACE" get deployment/cafe-assistant-api-stable >/dev/null
kubectl -n "$NAMESPACE" get deployment/cafe-assistant-api-canary >/dev/null
kubectl -n "$NAMESPACE" get service/cafe-assistant-api >/dev/null

if [[ -n "$CANARY_URL" ]]; then
  curl -fsS "${CANARY_URL%/}/health" >/dev/null
  TARGET_URL="$CANARY_URL" TENANT_ID="$TENANT_ID" uv run python deploy/shadow_traffic.py
else
  echo "WARNING: CANARY_URL was not set; promotion assumes shadow traffic already passed."
fi

kubectl -n "$NAMESPACE" set image deployment/cafe-assistant-api-stable "api=$IMAGE"
kubectl -n "$NAMESPACE" scale deployment/cafe-assistant-api-stable --replicas="$STABLE_REPLICAS"
kubectl -n "$NAMESPACE" rollout status deployment/cafe-assistant-api-stable --timeout=180s
kubectl -n "$NAMESPACE" scale deployment/cafe-assistant-api-canary --replicas=0

echo "Promoted $IMAGE to stable and scaled canary to zero."
