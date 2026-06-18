#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-cafe-assistant}"
IMAGE="${IMAGE:?set IMAGE promoted from canary}"
STABLE_REPLICAS="${STABLE_REPLICAS:-3}"

kubectl -n "$NAMESPACE" set image deployment/cafe-assistant-api-stable "api=$IMAGE"
kubectl -n "$NAMESPACE" rollout status deployment/cafe-assistant-api-stable --timeout=180s
kubectl -n "$NAMESPACE" scale deployment/cafe-assistant-api-stable --replicas="$STABLE_REPLICAS"
kubectl -n "$NAMESPACE" scale deployment/cafe-assistant-api-canary --replicas=0

echo "Promoted $IMAGE to stable and scaled canary to zero."
