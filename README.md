# Cafe Assistant

Production-oriented Python backend for a cafe menu assistant with deterministic
allergen and dietary safety filtering, hybrid menu retrieval, a safety-gated
streaming chat agent, consent-gated durable memory, tenant isolation, audit
logging, observability, automated evals, and release artifacts.

The most important rule in this system is simple:

> The model never decides dietary or allergen safety. It only explains or ranks
> items that have already passed deterministic safety checks.

## Contents

- [System Status](#system-status)
- [Architecture](#architecture)
- [Safety Model](#safety-model)
- [Repository Layout](#repository-layout)
- [Technology Stack](#technology-stack)
- [Local Development](#local-development)
- [Configuration](#configuration)
- [Database](#database)
- [Seed Data and Embeddings](#seed-data-and-embeddings)
- [API Surface](#api-surface)
- [Chat Agent Flow](#chat-agent-flow)
- [Identity and Memory](#identity-and-memory)
- [Security and Governance](#security-and-governance)
- [Observability](#observability)
- [Evaluation Suite](#evaluation-suite)
- [Testing and Quality Gates](#testing-and-quality-gates)
- [Deployment](#deployment)
- [Operations](#operations)
- [Development Notes](#development-notes)

## System Status

Implemented phases:

| Phase | Area | Status |
| --- | --- | --- |
| 0 | FastAPI, config, async SQLAlchemy, Alembic, Postgres/pgvector, Redis, seed data | Implemented |
| 1 | Deterministic dietary/allergen safety filter | Implemented |
| 2 | Exact, fuzzy, vector, and hybrid menu retrieval | Implemented |
| 3 | Streaming chat agent with explicit state machine | Implemented |
| 4 | QR tenant context, device identity, OTP consent, durable profile memory | Implemented |
| 5 | Tenant scoping, rate limiting, injection defenses, audit, redaction | Implemented |
| 6 | Tracing, metrics, eval hard gates, incident replay, CI | Implemented |
| 7 | Version registry, deploy artifacts, load/chaos tests, runbook | Implemented |

Current default model and embedding providers are deterministic local adapters.
They are intentionally provider-agnostic and mockable so tests and safety evals
run without external model calls.

## Architecture

### Runtime Components

```text
Browser / client
  |
  | POST /chat, identity endpoints, metrics/replay
  v
FastAPI app
  |
  | tenant resolution, request id, rate limit
  v
Chat agent state machine
  |
  | classify -> retrieve -> filter -> recommend -> compose
  v
Deterministic safety filter
  |
  | safe items only
  v
Composer / chat provider

PostgreSQL 16 + pgvector:
  relational menu source of truth, embeddings, profiles, consent, audit events

Redis:
  session memory, OTP challenges, rate limits
```

### Package Map

| Path | Responsibility |
| --- | --- |
| `src/cafe_assistant/main.py` | FastAPI application factory and route registration |
| `src/cafe_assistant/config.py` | Environment-driven settings via `pydantic-settings` |
| `src/cafe_assistant/db/` | SQLAlchemy models, async engine/session, repositories |
| `src/cafe_assistant/domain/dietary.py` | Pure deterministic allergen and dietary safety filter |
| `src/cafe_assistant/retrieval/` | Embedding text, pgvector search, keyword search, hybrid fusion |
| `src/cafe_assistant/gateway/model_gateway.py` | Mockable embedding and chat provider abstractions |
| `src/cafe_assistant/agent/` | Router, typed tools, custom FSM, composer, prompts |
| `src/cafe_assistant/memory/` | Redis session memory, durable profile merge, write gate |
| `src/cafe_assistant/identity/` | QR tenant context, device token, OTP upgrade flow |
| `src/cafe_assistant/security/` | Rate limiting, prompt-injection neutralization, audit, redaction |
| `src/cafe_assistant/observability/` | Tracing, metrics, incident replay |
| `evals/` | Safety and quality evaluation datasets and runners |
| `deploy/` | Containerfile, production Compose, Kubernetes manifests, release scripts |
| `tests/` | Unit, integration, chaos, and load tests |
| `scripts/` | Seed, embedding backfill, retention cleanup, incident replay helpers |

### Request Path

1. `api/deps.py` resolves tenant context from either `tenant_id` or QR payload.
2. Redis-backed rate limits are checked by session id and client IP.
3. `agent/router.py` classifies the message using rules and a cheap model path.
4. `retrieval/hybrid.py` combines keyword and semantic retrieval.
5. `domain/dietary.py` filters candidates deterministically.
6. The composer receives only the safe item set.
7. The response streams through `POST /chat` as server-sent events.
8. Significant actions are redacted and appended to `audit_events`.
9. Metrics and trace attributes are recorded for debugging and release gates.

## Safety Model

Safety is intentionally deterministic and conservative.

Hard invariants:

- No LLM call decides allergen or dietary safety.
- No unsafe item should appear in the model context.
- Unknown allergen data is unsafe when the customer has any active allergen
  avoidance.
- `menu_items.allergen_data_complete` is explicit. The system never infers
  completeness from ingredient rows.
- Allergen avoidances and dietary modes are hard exclusions.
- Sugar and carb preferences are soft ranking signals only. They never exclude
  items and are never medical advice.
- Medical questions are refused/escalated with a short not-medical-advice note.
- An empty safe set is valid. The assistant must say it cannot confirm a safe
  option and suggest checking with staff.

The core safety function is:

```python
filter_safe_items(items, restrictions) -> FilterResult
```

It returns both `safe_items` and per-item decisions such as:

- `INCLUDED`
- `EXCLUDED_ALLERGEN_PEANUT`
- `EXCLUDED_INCOMPLETE_DATA`
- `EXCLUDED_NOT_VEGAN`
- `EXCLUDED_NOT_VEGETARIAN`
- `EXCLUDED_NOT_GLUTEN_FREE`

## Repository Layout

```text
.
|-- alembic.ini
|-- docker-compose.yml
|-- pyproject.toml
|-- README.md
|-- RUNBOOK.md
|-- deploy/
|-- evals/
|-- migrations/
|-- scripts/
|-- src/
|   `-- cafe_assistant/
|-- static/
|-- tests/
`-- uv.lock
```

Generated or local-only files such as `.env`, `.venv/`, `__pycache__/`, and test
caches should not be committed.

## Technology Stack

- Python 3.12
- FastAPI
- SQLAlchemy 2.0 async ORM, declarative typed models
- Alembic
- PostgreSQL 16 with `pgvector`
- Redis
- pydantic-settings
- pytest, pytest-asyncio, httpx
- Ruff
- Docker Compose for local infrastructure
- Kubernetes and production Compose deployment artifacts

## Local Development

### Prerequisites

- Python 3.12
- `uv`
- Docker Desktop or compatible Docker runtime
- PowerShell, Bash, or another shell capable of running the commands below

### 1. Create Environment File

```bash
cp .env.example .env
```

On PowerShell:

```powershell
Copy-Item .env.example .env
```

For local development, the default `.env.example` values are enough.

### 2. Install Dependencies

```bash
uv sync --extra dev
```

This creates or updates the local virtual environment. Do not commit `.venv/`.

### 3. Start Local Infrastructure

```bash
docker compose up -d
```

This starts:

- PostgreSQL 16 with `pgvector`
- Redis 7

Check rendered Compose config:

```bash
docker compose config
```

### 4. Run Migrations

```bash
uv run alembic upgrade head
```

To render SQL without applying it:

```bash
uv run alembic upgrade head --sql
```

### 5. Seed Menu Data

```bash
uv run python scripts/seed_menu.py
```

### 6. Backfill Embeddings

```bash
uv run python scripts/embed_menu.py
```

### 7. Run the API

```bash
uv run uvicorn cafe_assistant.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Open the minimal chat page:

```text
http://localhost:8000/chat
```

## Configuration

Settings are loaded from environment variables by `src/cafe_assistant/config.py`.
The app also reads `.env` locally.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `Cafe Assistant` | FastAPI application name |
| `ENVIRONMENT` | `local` | Runtime environment label |
| `DATABASE_URL` | local Postgres URL | Async SQLAlchemy database URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `EMBEDDING_PROVIDER` | `hash` | Embedding provider adapter |
| `EMBEDDING_DIMENSION` | `8` | Embedding vector dimension |
| `CHEAP_CHAT_PROVIDER` | `local` | Cheap model path for classification |
| `STRONG_CHAT_PROVIDER` | `local` | Strong model path for synthesis |
| `CHAT_TIMEOUT_SECONDS` | `8.0` | Per-provider chat timeout |
| `CHAT_RETRIES` | `1` | Provider retry count |
| `AGENT_DEADLINE_SECONDS` | `12.0` | Per-request agent deadline |
| `AGENT_MAX_TOOL_CALLS` | `4` | Tool-call budget |
| `IDENTITY_PHONE_HASH_SECRET` | local secret | Salt/secret for phone hashing |
| `DEVICE_TOKEN_BYTES` | `32` | Opaque browser token size |
| `OTP_CODE_TTL_SECONDS` | `300` | OTP challenge TTL |
| `RATE_LIMIT_SESSION_REQUESTS` | `60` | Session request limit |
| `RATE_LIMIT_SESSION_WINDOW_SECONDS` | `60` | Session rate-limit window |
| `RATE_LIMIT_IP_REQUESTS` | `120` | IP request limit |
| `RATE_LIMIT_IP_WINDOW_SECONDS` | `60` | IP rate-limit window |
| `PROFILE_RETENTION_DAYS` | `365` | Durable profile retention window |
| `SESSION_RETENTION_DAYS` | `14` | Session/event retention window |
| `AUDIT_RETENTION_DAYS` | `730` | Audit retention window |
| `LANGFUSE_ENABLED` | `false` | Enable Langfuse integration |
| `LANGFUSE_PUBLIC_KEY` | empty | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | empty | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse cloud URL | Langfuse host |
| `DEFAULT_CHAT_MODEL_NAME` | `local` | Trace label for default model |
| `LATENCY_BUDGET_MS` | `1500` | Eval/load latency budget |

Production environments must replace local secrets and should provide
configuration through environment variables or secret managers, not committed
files.

## Database

The relational database is the source of truth. The vector store is only an
index over menu data.

Core schema:

- `tenants`
- `locations`
- `menu_items`
- `ingredients`
- `item_ingredients`
- `allergens`
- `ingredient_allergens`
- `dietary_tags`
- `item_dietary_tags`
- `customers`
- `customer_profile`
- `episodic_events`
- `consents`
- `device_tokens`
- `audit_events`

Important menu columns:

- `menu_items.embedding`: pgvector embedding for semantic retrieval.
- `menu_items.allergen_data_complete`: explicit boolean safety flag. If false
  and the customer has allergen avoidance, the item is excluded.

Migration chain:

| Revision | Purpose |
| --- | --- |
| `20260616_0001_initial_schema.py` | Initial menu schema and pgvector extension |
| `20260617_0002_menu_item_embeddings.py` | Menu item embedding column |
| `20260618_0003_identity_memory.py` | Customer, profile, consent, episodic memory, device token tables |
| `20260618_0004_security_audit.py` | Append-only audit event table |

Run migrations:

```bash
uv run alembic upgrade head
```

Create a new migration after model changes:

```bash
uv run alembic revision --autogenerate -m "describe change"
```

Always inspect generated migrations before applying them.

## Seed Data and Embeddings

Seed the deterministic demo menu:

```bash
uv run python scripts/seed_menu.py
```

The seed data inserts one tenant, one location, and realistic cafe items such as
coffees, teas, pastries, and sandwiches. It includes ingredients, allergens,
dietary tags, sugar, carb data, and intentionally incomplete allergen data for
some items.

Backfill embeddings:

```bash
uv run python scripts/embed_menu.py
```

Embedding text is built from menu item name, description, and tags. The default
hash provider is deterministic and local. Real provider adapters can be added
behind the existing `EmbeddingProvider` protocol.

## API Surface

### Health

```http
GET /health
```

Returns:

```json
{"status":"ok"}
```

### Chat

```http
POST /chat
```

Request body:

```json
{
  "session_id": "demo-session",
  "tenant_id": 1,
  "device_token": null,
  "message": "I am allergic to peanuts. What pastry can I have?"
}
```

The endpoint streams server-sent events:

```text
data: {"token":"..."}

event: done
data: {}
```

Response headers include:

- `X-Request-ID`
- `X-Trace-ID`

### Chat Page

```http
GET /chat
```

Serves `static/chat.html`, a minimal vanilla JavaScript chat client.

### Identity and Consent

```http
POST /identity/otp/start
POST /identity/otp/confirm
GET /identity/profile
DELETE /identity/profile
```

The OTP flow upgrades an anonymous session to a remembered profile only after
explicit customer consent. Profile read and deletion are tenant scoped.

### Observability

```http
GET /metrics
GET /observability/replay/{trace_id}
```

`/metrics` returns in-process reliability, quality, latency, and estimated cost
metrics. Replay reconstructs stored trace details for incident debugging.

## Chat Agent Flow

The agent is a single custom finite state machine with these states:

```text
CLASSIFIED -> RETRIEVING -> FILTERING -> RECOMMENDING -> COMPOSING -> COMPLETE
```

Additional terminal states:

- `ESCALATED`
- `FAILED`

The state machine enforces:

- per-request deadline
- maximum tool-call budget
- medical refusal path
- empty-safe-set fallback
- memory-unavailable fallback
- recommender-unavailable fallback

Tools are typed and exposed through `agent/tools.py`:

- `menu_lookup`
- `search_menu`
- `dietary_filter`

The composer receives only `safe_items`. It never receives the raw menu.

## Identity and Memory

Identity degrades gracefully:

1. Device token hit: recognized customer profile is loaded.
2. Device token miss: request continues as an anonymous session.
3. Redis/session unavailable: request continues as an anonymous session.

Tenant context is stamped from either:

- a direct `tenant_id`, useful for development and tests
- a QR payload containing only cafe/location/table context

QR payloads must never contain user identity or secrets.

Durable memory rules:

- UI preferences, such as milk preference, may be auto-written for recognized
  customers.
- Dietary or health facts, such as allergies or diabetic mentions, require
  explicit OTP consent before durable persistence.
- Current message instructions override stored memory for the active turn.
- Customer profiles are tenant scoped. Cross-tenant reads are denied by design.

## Security and Governance

Security controls:

- Every data-bearing request requires tenant context.
- All data access is tenant scoped in repositories and API dependencies.
- Redis-backed rate limits apply per session and per IP.
- User text and menu text are treated as untrusted.
- Prompt-injection-like phrases are neutralized before model/provider calls.
- System instructions and retrieved data are kept separated.
- Logs and audit payloads redact phone numbers, health terms, tokens, and
  secrets.
- Significant actions append redacted rows to `audit_events`.
- App code provides no update/delete path for audit events.

Governed actions written to audit:

- recommendation served
- profile read
- profile write
- consent granted
- profile deleted

Retention cleanup:

```bash
uv run python scripts/cleanup_retention.py --dry-run
uv run python scripts/cleanup_retention.py
```

## Observability

Tracing records request and model-related attributes including:

- tenant id
- request id
- trace id
- route and classification confidence
- prompt version
- tool name
- retrieved item ids
- safe item ids
- model/provider name
- token/cost estimates where available
- component version registry

The version registry tracks:

- prompts
- tools
- retrievers
- embedding model
- model choices
- policy rules
- memory write rules
- orchestrator graph

Replay a trace through the API:

```bash
curl http://localhost:8000/observability/replay/<trace_id>
```

Replay a trace from the command line:

```bash
uv run python scripts/incident_replay.py <trace_id>
```

Langfuse is configurable through environment variables and disabled by default.
Tests run against no-op/local observability behavior.

## Evaluation Suite

The evaluation suite contains realistic and adversarial cases:

- allergen avoidance
- semantically similar unsafe items
- incomplete allergen data
- vegan, vegetarian, and gluten-free modes
- conflicting user instructions
- prompt-injection strings
- medical questions
- empty-safe-set situations
- groundedness checks
- relevance checks
- latency budget checks

Run all evals:

```bash
uv run python evals/run_evals.py
```

Run the hard allergen-safety gate only:

```bash
uv run python evals/allergen_safety.py
```

The allergen safety gate fails if false negatives are greater than zero.

## Testing and Quality Gates

Run the full test suite:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check . --no-cache
```

Run load tests included in pytest:

```bash
uv run pytest tests/load -q
```

Run chaos tests:

```bash
uv run pytest tests/chaos -q
```

Optional k6 script:

```bash
k6 run tests/load/k6_chat.js
```

Expected release checks:

```bash
uv run ruff check . --no-cache
uv run pytest -q
uv run python evals/run_evals.py
uv run alembic upgrade head --sql
docker compose config
```

The CI workflow runs lint, tests, and evals. Allergen safety is a hard gate.

## Deployment

Deployment artifacts live in `deploy/`.

### Container

```bash
docker build -f deploy/Containerfile -t cafe-assistant:local .
```

### Production Compose

Production Compose requires a non-empty Postgres password:

```bash
POSTGRES_PASSWORD=replace-me docker compose -f deploy/docker-compose.prod.yml config
POSTGRES_PASSWORD=replace-me docker compose -f deploy/docker-compose.prod.yml up -d
```

Production Compose includes:

- API service
- retention cleanup worker
- Postgres with pgvector
- Redis
- health checks
- resource limits

### Kubernetes

Manifests live in `deploy/k8s/`:

- namespace
- config map
- secret example
- Postgres and Redis
- stable and canary API deployments
- service
- cleanup worker

Apply manifests:

```bash
kubectl apply -f deploy/k8s/
```

### Release, Canary, Promotion, Rollback

Start canary:

```bash
IMAGE=ghcr.io/org/cafe-assistant:<tag> deploy/release.sh
```

Run shadow traffic:

```bash
TARGET_URL=http://<canary-url> TENANT_ID=1 uv run python deploy/shadow_traffic.py
```

Promote:

```bash
IMAGE=ghcr.io/org/cafe-assistant:<tag> deploy/promote.sh
```

Rollback:

```bash
deploy/rollback.sh
```

See `RUNBOOK.md` for the full incident and release playbook.

## Operations

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### Trace Replay

```bash
curl http://localhost:8000/observability/replay/<trace_id>
```

### Retention Cleanup

```bash
uv run python scripts/cleanup_retention.py --dry-run
uv run python scripts/cleanup_retention.py
```

### Incident Response

Use `RUNBOOK.md` for:

- how to read a trace
- how to replay an incident
- safety incident triage
- rollback steps
- launch gate checklist
- ownership/page guidance

## Development Notes

- Keep business logic out of migrations and scripts unless it is one-time data
  setup.
- Keep the safety filter pure, deterministic, and free of network calls.
- Keep model providers behind gateway protocols so tests can inject fakes.
- Do not let LLM output bypass `filter_safe_items`.
- Do not persist health or dietary facts without consent.
- Do not log raw phone numbers, health facts, tokens, or secrets.
- Do not commit `.env`, `.venv/`, database volumes, caches, or generated local
  files.

Before opening a pull request or pushing a release branch, run:

```bash
uv run ruff check . --no-cache
uv run pytest -q
uv run python evals/run_evals.py
```
