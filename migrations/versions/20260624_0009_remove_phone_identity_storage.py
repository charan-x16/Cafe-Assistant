"""Remove legacy phone-hash identity storage.

Revision ID: 20260624_0009
Revises: 20260624_0008
Create Date: 2026-06-24 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260624_0009"
down_revision: str | None = "20260624_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop legacy phone-hash identity columns and indexes.

    Args:
        None:
            Alembic supplies the migration context.

    Returns:
        None:
            The database schema is updated in place so customers authenticate
            only through tenant-scoped username/password accounts.
    """
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("customers") as batch_op:
            batch_op.drop_index("ix_customers_phone_hash")
            batch_op.drop_column("phone_hash")
        return

    op.drop_index(op.f("ix_customers_phone_hash"), table_name="customers")
    op.drop_constraint("uq_customers_tenant_phone_hash", "customers", type_="unique")
    op.drop_column("customers", "phone_hash")


def downgrade() -> None:
    """Restore legacy phone-hash identity storage.

    Args:
        None:
            Alembic supplies the migration context.

    Returns:
        None:
            The database schema is reverted for this revision.
    """
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("customers") as batch_op:
            batch_op.add_column(sa.Column("phone_hash", sa.String(length=128), nullable=True))
            batch_op.create_index("ix_customers_phone_hash", ["phone_hash"], unique=False)
        return

    op.add_column("customers", sa.Column("phone_hash", sa.String(length=128), nullable=True))
    op.create_unique_constraint(
        "uq_customers_tenant_phone_hash",
        "customers",
        ["tenant_id", "phone_hash"],
    )
    op.create_index(op.f("ix_customers_phone_hash"), "customers", ["phone_hash"], unique=False)