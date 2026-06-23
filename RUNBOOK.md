# Cafe Assistant Runbook

## Service Summary

The cafe assistant is a FastAPI backend with deterministic dietary/allergen
safety filtering, hybrid menu retrieval, session memory, consent-gated durable
profiles, audit logging, tracing, and automated safety evals.

Production runtime components:

- API: `cafe_assistant.main:app`
- Worker: retention cleanup via `scripts/cleanup_retention.py`
- Database: PostgreSQL 16 with pgvector
- Cache/session/rate-limit store: Redis

## How to Read a Trace

Use the secured replay endpoint for deployed traces:

```bash
curl -H "X-Admin-Token: $OBSERVABILITY_ADMIN_TOKEN" \
  "$BASE_URL/observability/replay/$TRACE_ID?tenant_id=$TENANT_ID"
```

Use the local durable JSONL trace spool from an operator shell:

```bash
uv run python scripts/incident_replay.py "$TRACE_ID"
```

Or ask the CLI to call the secured API for you:

```bash
uv run python scripts/incident_replay.py "$TRACE_ID" \
  --base-url "$BASE_URL" \
  --tenant-id "$TENANT_ID" \
  --admin-token "$OBSERVABILITY_ADMIN_TOKEN"
```

Look for:

- `version_registry`: prompt, tool, retriever, model, policy, memory-rule, and
  orchestrator versions active for the request.
- `route` and `route_confidence`: classifier decision used before retrieval.
- `prompt_context`: sanitized model messages and prompt version.
- `retrieved_items`: candidate item ids returned by retrieval/tool spans.
- `tools`: `menu_lookup`, `search_menu`, and `dietary_filter` spans.
- `spans`: span IDs, parent span IDs, duration, errors, and redacted attributes
  for each request step.

For a safety incident, confirm that the LLM context contains only `SAFE_ITEM`
lines and that unsafe/allergen-incomplete items were excluded before composing.

## Incident Replay

1. Find `trace_id` from the response headers, audit event, or redacted logs.
2. Replay through the API when debugging a deployed request, or through the local
   durable spool when working from the same persistent trace volume.
3. Capture:
   - request id and tenant id
   - version registry
   - route and classifier confidence
   - retrieved item ids
   - safe item ids
   - prompt version and sanitized prompt context
   - any failed tool, retrieval, observability, or model spans
4. Re-run the eval gate locally:

```bash
uv run python evals/run_evals.py --strict
```

5. If allergen false negatives are nonzero, freeze rollout and roll back. If strict mode reports another family failure, keep the rollout frozen until the failing dataset is understood or relabeled.

## Rollback

Kubernetes one-command rollback:

```bash
deploy/rollback.sh
```

This runs `kubectl rollout undo` for the stable API deployment, waits for rollout
health, and scales canary to zero.

For production Compose:

```bash
CAFE_ASSISTANT_IMAGE="$LAST_KNOWN_GOOD_IMAGE" docker compose -f deploy/docker-compose.prod.yml up -d
```

Always verify after rollback:

```bash
curl "$BASE_URL/health"
uv run python evals/allergen_safety.py
```

## Canary and Shadow Traffic

Start canary and run remote shadow checks when a canary URL is available:

```bash
IMAGE="$NEW_IMAGE" CANARY_REPLICAS=1 CANARY_URL="$CANARY_URL" TENANT_ID=1 deploy/release.sh
```

Run shadow traffic manually if `CANARY_URL` was not supplied to `deploy/release.sh`:

```bash
TARGET_URL="$CANARY_URL" TENANT_ID=1 uv run python deploy/shadow_traffic.py
```

Promote only after shadow traffic, strict evals, and required load checks pass:

```bash
IMAGE="$NEW_IMAGE" CANARY_URL="$CANARY_URL" TENANT_ID=1 deploy/promote.sh
```
Abort:

```bash
deploy/rollback.sh
```

## Who to Page

- Safety/allergen incident: on-call backend owner and product safety owner.
- Database outage or migration issue: backend owner and platform/database owner.
- Redis/rate-limit/session outage: platform owner; backend owner verifies
  anonymous fallback.
- LLM/provider degradation: backend owner; assistant should continue using safe
  fallback behavior or local provider fallback.

## Safe Degradation Expectations

- Recommender failure: return safety-filtered popular/menu items, never raw menu
  items.
- Allergen data unavailable or incomplete with active allergen avoidance: say the
  assistant cannot confirm a safe option and ask the customer to check with staff.
- Memory unavailable: continue as a functional anonymous session.
- Medical questions: refuse with the not-medical-advice note.

## Launch Gate Checklist

- Migrations render: `uv run alembic upgrade head --sql`
  Acceptance: command exits zero and includes all revisions through head.
- Unit/integration/chaos/load tests: `uv run pytest -q`
  Acceptance: all tests pass; p95 first-token latency below 2 seconds and p99
  below 2.5 seconds in the CI load smoke.
- Lint: `uv run ruff check . --no-cache`
  Acceptance: exits zero.
- Safety evals: `uv run python evals/run_evals.py --strict`
  Acceptance: allergen false-negative count is exactly zero; empty-safe-set, groundedness, relevance, medical-refusal, and latency families all pass.
- Standalone hard gate: `uv run python evals/allergen_safety.py`
  Acceptance: exits zero with `false_negative_count=0`.
- Compose validation: `docker compose config` and
  `docker compose -f deploy/docker-compose.prod.yml config` with required production env placeholders.
  Acceptance: both render valid config and production config resolves BGE, Qdrant, OpenAI, and required secrets.
- Canary release: `IMAGE=$NEW_IMAGE CANARY_URL=$CANARY_URL deploy/release.sh`
  Acceptance: canary rollout reaches ready state, health passes, and shadow traffic passes against legacy plus BTB adversarial cases.
- Rollback drill: `deploy/rollback.sh`
  Acceptance: stable deployment returns to last known good image and health checks
  pass.
- Observability: `GET /metrics`, `GET /metrics/openmetrics`, and `GET /observability/replay/{trace_id}`
  Acceptance: metrics families are present, OpenMetrics renders, replay is tenant/admin scoped, and traces include the version registry.
