# Cafe Assistant

Backend foundations for a cafe menu assistant. This repository currently contains
only infrastructure, configuration, schema, migrations, seed data, and a health
endpoint. There is no chat or AI logic yet.

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
python -m pip install -e ".[dev]"
```

Start PostgreSQL and Redis:

```bash
docker compose up -d
```

Run database migrations:

```bash
alembic upgrade head
```

Seed the sample menu:

```bash
python scripts/seed_menu.py
```

Run the API:

```bash
uvicorn cafe_assistant.main:app --reload
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
pytest
```

## Data Notes

Menu item allergen coverage is tracked explicitly with
`menu_items.allergen_data_complete`. The seed data intentionally leaves some
items incomplete, with missing ingredient allergen mappings, so downstream code
can treat unknown allergen data as unsafe.
