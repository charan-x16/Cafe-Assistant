#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-cafe-assistant}"

kubectl -n "$NAMESPACE" get deployment/cafe-assistant-api-stable >/dev/null
kubectl -n "$NAMESPACE" rollout undo deployment/cafe-assistant-api-stable
kubectl -n "$NAMESPACE" rollout status deployment/cafe-assistant-api-stable --timeout=180s
kubectl -n "$NAMESPACE" scale deployment/cafe-assistant-api-canary --replicas=0

echo "Rollback complete; canary traffic disabled. Verify /health on the stable service."
