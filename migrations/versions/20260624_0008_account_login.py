"""Add tenant-scoped username password accounts.

Revision ID: 20260624_0008
Revises: 20260622_0007
Create Date: 2026-06-24 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260624_0008"
down_revision: str | None = "20260622_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable account credentials to existing customer rows.

    Args:
        None:
            Alembic supplies the migration context.

    Returns:
        None:
            The database schema is updated in place.
    """
    op.add_column("customers", sa.Column("username", sa.String(length=150), nullable=True))
    op.add_column("customers", sa.Column("password_hash", sa.String(length=512), nullable=True))
    op.create_index(op.f("ix_customers_username"), "customers", ["username"], unique=False)
    op.create_index(
        "uq_customers_tenant_username",
        "customers",
        ["tenant_id", "username"],
        unique=True,
        postgresql_where=sa.text("username IS NOT NULL"),
        sqlite_where=sa.text("username IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove account credential columns from customer rows.

    Args:
        None:
            Alembic supplies the migration context.

    Returns:
        None:
            The database schema is reverted for this revision.
    """
    op.drop_index("uq_customers_tenant_username", table_name="customers")
    op.drop_index(op.f("ix_customers_username"), table_name="customers")
    op.drop_column("customers", "password_hash")
    op.drop_column("customers", "username")