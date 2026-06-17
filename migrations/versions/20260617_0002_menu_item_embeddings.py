"""Add menu item embeddings.

Revision ID: 20260617_0002
Revises: 20260616_0001
Create Date: 2026-06-17 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

revision: str = "20260617_0002"
down_revision: str | None = "20260616_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("ALTER TABLE menu_items ADD COLUMN embedding vector(8)")
    op.execute(
        """
        CREATE INDEX ix_menu_items_embedding_hnsw
        ON menu_items
        USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX ix_menu_items_search_tsv
        ON menu_items
        USING gin (
            to_tsvector(
                'english',
                coalesce(name, '') || ' ' ||
                coalesce(description, '') || ' ' ||
                coalesce(category, '')
            )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX ix_menu_items_name_trgm
        ON menu_items
        USING gin (name gin_trgm_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_menu_items_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_menu_items_search_tsv")
    op.execute("DROP INDEX IF EXISTS ix_menu_items_embedding_hnsw")
    op.execute("ALTER TABLE menu_items DROP COLUMN IF EXISTS embedding")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
