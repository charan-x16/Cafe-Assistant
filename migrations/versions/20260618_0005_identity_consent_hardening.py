"""Harden identity consent and device token storage.

Revision ID: 20260618_0005
Revises: 20260618_0004
Create Date: 2026-06-18 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260618_0005"
down_revision: str | None = "20260618_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add token lifecycle fields and active-consent uniqueness.

    Args:
        None.

    Returns:
        None:
            Alembic applies schema changes to the connected database.
    """
    op.add_column(
        "customer_device_tokens",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "customer_device_tokens",
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        UPDATE customer_device_tokens
        SET expires_at = COALESCE(created_at, NOW()) + INTERVAL '90 days'
        WHERE expires_at IS NULL
        """
    )
    op.alter_column("customer_device_tokens", "expires_at", nullable=False)
    op.create_index(
        "uq_consents_active_customer_scope",
        "consents",
        ["customer_id", "scope"],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )


def downgrade() -> None:
    """Remove token lifecycle fields and active-consent uniqueness.

    Args:
        None.

    Returns:
        None:
            Alembic reverts schema changes from this revision.
    """
    op.drop_index("uq_consents_active_customer_scope", table_name="consents")
    op.drop_column("customer_device_tokens", "revoked_at")
    op.drop_column("customer_device_tokens", "expires_at")
