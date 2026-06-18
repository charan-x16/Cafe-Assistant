#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-cafe-assistant}"
IMAGE="${IMAGE:?set IMAGE, e.g. ghcr.io/org/cafe-assistant:2026-06-18}"
CANARY_REPLICAS="${CANARY_REPLICAS:-1}"
STABLE_REPLICAS="${STABLE_REPLICAS:-3}"

kubectl -n "$NAMESPACE" set image deployment/cafe-assistant-api-canary "api=$IMAGE"
kubectl -n "$NAMESPACE" scale deployment/cafe-assistant-api-canary --replicas="$CANARY_REPLICAS"
kubectl -n "$NAMESPACE" rollout status deployment/cafe-assistant-api-canary --timeout=180s

echo "Canary is live. Run shadow traffic and eval checks before promotion:"
echo "  TARGET_URL=http://<service> uv run python deploy/shadow_traffic.py"
echo "Promote when ready:"
echo "  IMAGE=$IMAGE deploy/promote.sh"
echo "Rollback anytime:"
echo "  deploy/rollback.sh"
