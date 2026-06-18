# Cafe Assistant

Backend for a cafe menu assistant with deterministic dietary/allergen safety,
hybrid menu retrieval, streaming chat, and consent-gated durable memory.

## Stack

- Python 3.12
- FastAPI
- SQLAlchemy 2.0 async ORM
- Alembic
- PostgreSQL 16 with pgvector
- Redis
- pydantic-settings

## Setup

Create local environment variables:

```bash
cp .env.example .env
```

Install the project with development dependencies:

```bash
uv sync --extra dev
```

Start PostgreSQL and Redis:

```bash
docker compose up -d
```

Run database migrations:

```bash
uv run alembic upgrade head
```

Seed the sample menu:

```bash
uv run python scripts/seed_menu.py
uv run python scripts/embed_menu.py
```

Run the API:

```bash
uv run uvicorn cafe_assistant.main:app --reload
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check . --no-cache
```

## Chat and Memory

- `GET /health` returns `{"status":"ok"}`.
- `POST /chat` streams SSE tokens. Send `tenant_id` for local development or a
  QR payload containing only `cafe_id`, `location_id`, and `table_id`.
- `GET /chat` serves the minimal browser chat page.
- `POST /identity/otp/start` and `POST /identity/otp/confirm` upgrade an
  anonymous session to a remembered profile after explicit OTP consent.
- `GET /identity/profile` inspects the recognized profile by tenant-scoped
  device token.
- `DELETE /identity/profile` deletes the profile, consent, events, and device
  token mapping.

## Data Notes

Menu item allergen coverage is tracked explicitly with
`menu_items.allergen_data_complete`. The seed data intentionally leaves some
items incomplete, with missing ingredient allergen mappings, so downstream code
can treat unknown allergen data as unsafe.

Health and dietary facts are durable only after the OTP consent flow grants the
`dietary_health` scope. UI preferences such as a milk preference may be saved
automatically for a recognized customer. Current-turn instructions still
override stored profile memory for that active chat turn.

## Security and Governance

- Data endpoints resolve tenant context through shared API dependencies and apply
  Redis-backed per-session and per-IP rate limits.
- User text and menu text are treated as untrusted before model/provider calls;
  instruction-like phrases are neutralized and model context is guarded.
- Significant actions write redacted audit events to `audit_events`, including
  recommendations, profile reads/writes, consent grants, and profile deletion.
- Logs and audit payloads redact phone numbers, health terms, tokens, and
  secrets.
- Run retention cleanup with:

```bash
uv run python scripts/cleanup_retention.py --dry-run
```

## Observability and Evals

- `GET /metrics` returns in-process reliability, quality, latency, and estimated
  cost metrics.
- `GET /observability/replay/{trace_id}` reconstructs stored trace details,
  including prompt context, retrieved item ids, tool spans, and prompt versions.
- Langfuse tracing is configurable with `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`; it is disabled by default.

Run the full eval suite:

```bash
uv run python evals/run_evals.py
```

Run the hard allergen-safety gate only:

```bash
uv run python evals/allergen_safety.py
```

CI runs lint, tests, and `evals/run_evals.py`; the build fails if the allergen
false-negative count is greater than zero.
