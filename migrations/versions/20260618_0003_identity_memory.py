"""Add identity and durable memory tables.

Revision ID: 20260618_0003
Revises: 20260617_0002
Create Date: 2026-06-18 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260618_0003"
down_revision: str | None = "20260617_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("phone_hash", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_customers_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_customers")),
        sa.UniqueConstraint(
            "tenant_id",
            "phone_hash",
            name="uq_customers_tenant_phone_hash",
        ),
    )
    op.create_index(op.f("ix_customers_phone_hash"), "customers", ["phone_hash"], unique=False)
    op.create_index(op.f("ix_customers_tenant_id"), "customers", ["tenant_id"], unique=False)

    op.create_table(
        "customer_profile",
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column(
            "preferences",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "dietary_facts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("consent_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name=op.f("fk_customer_profile_customer_id_customers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("customer_id", name=op.f("pk_customer_profile")),
    )

    op.create_table(
        "consents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(length=100), nullable=False),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name=op.f("fk_consents_customer_id_customers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consents")),
    )
    op.create_index(op.f("ix_consents_customer_id"), "consents", ["customer_id"], unique=False)
    op.create_index(op.f("ix_consents_scope"), "consents", ["scope"], unique=False)

    op.create_table(
        "customer_device_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name=op.f("fk_customer_device_tokens_customer_id_customers"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_customer_device_tokens_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_customer_device_tokens")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_customer_device_tokens_token_hash")),
    )
    op.create_index(
        op.f("ix_customer_device_tokens_customer_id"),
        "customer_device_tokens",
        ["customer_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_customer_device_tokens_tenant_id"),
        "customer_device_tokens",
        ["tenant_id"],
        unique=False,
    )

    op.create_table(
        "episodic_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name=op.f("fk_episodic_events_customer_id_customers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_episodic_events")),
    )
    op.create_index(
        op.f("ix_episodic_events_customer_id"),
        "episodic_events",
        ["customer_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_episodic_events_customer_id"), table_name="episodic_events")
    op.drop_table("episodic_events")
    op.drop_index(op.f("ix_customer_device_tokens_tenant_id"), table_name="customer_device_tokens")
    op.drop_index(
        op.f("ix_customer_device_tokens_customer_id"),
        table_name="customer_device_tokens",
    )
    op.drop_table("customer_device_tokens")
    op.drop_index(op.f("ix_consents_scope"), table_name="consents")
    op.drop_index(op.f("ix_consents_customer_id"), table_name="consents")
    op.drop_table("consents")
    op.drop_table("customer_profile")
    op.drop_index(op.f("ix_customers_tenant_id"), table_name="customers")
    op.drop_index(op.f("ix_customers_phone_hash"), table_name="customers")
    op.drop_table("customers")
