"""Use 384-dimensional sentence-transformer embeddings.

Revision ID: 20260619_0006
Revises: 20260619_0005
Create Date: 2026-06-19 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

revision: str = "20260619_0006"
down_revision: str | None = "20260619_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this Alembic migration to the database schema.

    Args:
        None.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    op.execute("DROP INDEX IF EXISTS ix_menu_items_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_catalog_item_embeddings_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_policy_chunk_embeddings_embedding_hnsw")

    op.execute("UPDATE menu_items SET embedding = NULL")
    op.execute("DELETE FROM catalog_item_embeddings")
    op.execute("DELETE FROM policy_chunk_embeddings")

    op.execute("ALTER TABLE menu_items ALTER COLUMN embedding TYPE vector(384)")
    op.execute("ALTER TABLE catalog_item_embeddings ALTER COLUMN embedding TYPE vector(384)")
    op.execute("ALTER TABLE policy_chunk_embeddings ALTER COLUMN embedding TYPE vector(384)")

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
        CREATE INDEX ix_catalog_item_embeddings_embedding_hnsw
        ON catalog_item_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX ix_policy_chunk_embeddings_embedding_hnsw
        ON policy_chunk_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )


def downgrade() -> None:
    """Reverse this Alembic migration from the database schema.

    Args:
        None.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    op.execute("DROP INDEX IF EXISTS ix_menu_items_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_catalog_item_embeddings_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_policy_chunk_embeddings_embedding_hnsw")

    op.execute("UPDATE menu_items SET embedding = NULL")
    op.execute("DELETE FROM catalog_item_embeddings")
    op.execute("DELETE FROM policy_chunk_embeddings")

    op.execute("ALTER TABLE menu_items ALTER COLUMN embedding TYPE vector(8)")
    op.execute("ALTER TABLE catalog_item_embeddings ALTER COLUMN embedding TYPE vector(8)")
    op.execute("ALTER TABLE policy_chunk_embeddings ALTER COLUMN embedding TYPE vector(8)")

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
        CREATE INDEX ix_catalog_item_embeddings_embedding_hnsw
        ON catalog_item_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX ix_policy_chunk_embeddings_embedding_hnsw
        ON policy_chunk_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )
