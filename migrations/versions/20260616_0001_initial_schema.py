"""Initial schema.

Revision ID: 20260616_0001
Revises:
Create Date: 2026-06-16 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260616_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "allergens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_allergens")),
        sa.UniqueConstraint("code", name=op.f("uq_allergens_code")),
    )
    op.create_table(
        "dietary_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dietary_tags")),
        sa.UniqueConstraint("code", name=op.f("uq_dietary_tags_code")),
    )
    op.create_table(
        "ingredients",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingredients")),
        sa.UniqueConstraint("name", name=op.f("uq_ingredients_name")),
    )
    op.create_table(
        "tenants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tenants")),
        sa.UniqueConstraint("name", name=op.f("uq_tenants_name")),
    )
    op.create_table(
        "ingredient_allergens",
        sa.Column("ingredient_id", sa.Integer(), nullable=False),
        sa.Column("allergen_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["allergen_id"],
            ["allergens.id"],
            name=op.f("fk_ingredient_allergens_allergen_id_allergens"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["ingredient_id"],
            ["ingredients.id"],
            name=op.f("fk_ingredient_allergens_ingredient_id_ingredients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "ingredient_id",
            "allergen_id",
            name=op.f("pk_ingredient_allergens"),
        ),
    )
    op.create_table(
        "locations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_locations_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_locations")),
    )
    op.create_index(op.f("ix_locations_tenant_id"), "locations", ["tenant_id"], unique=False)
    op.create_table(
        "menu_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=False),
        sa.Column("price_cents", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("sugar_grams", sa.Numeric(precision=6, scale=1), nullable=True),
        sa.Column("carbs_grams", sa.Numeric(precision=6, scale=1), nullable=True),
        sa.Column("is_available", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "allergen_data_complete",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_menu_items_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_menu_items")),
    )
    op.create_index(op.f("ix_menu_items_tenant_id"), "menu_items", ["tenant_id"], unique=False)
    op.create_table(
        "item_dietary_tags",
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("dietary_tag_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dietary_tag_id"],
            ["dietary_tags.id"],
            name=op.f("fk_item_dietary_tags_dietary_tag_id_dietary_tags"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["menu_items.id"],
            name=op.f("fk_item_dietary_tags_item_id_menu_items"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("item_id", "dietary_tag_id", name=op.f("pk_item_dietary_tags")),
    )
    op.create_table(
        "item_ingredients",
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("ingredient_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["ingredient_id"],
            ["ingredients.id"],
            name=op.f("fk_item_ingredients_ingredient_id_ingredients"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["menu_items.id"],
            name=op.f("fk_item_ingredients_item_id_menu_items"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("item_id", "ingredient_id", name=op.f("pk_item_ingredients")),
    )


def downgrade() -> None:
    op.drop_table("item_ingredients")
    op.drop_table("item_dietary_tags")
    op.drop_index(op.f("ix_menu_items_tenant_id"), table_name="menu_items")
    op.drop_table("menu_items")
    op.drop_index(op.f("ix_locations_tenant_id"), table_name="locations")
    op.drop_table("locations")
    op.drop_table("ingredient_allergens")
    op.drop_table("tenants")
    op.drop_table("ingredients")
    op.drop_table("dietary_tags")
    op.drop_table("allergens")
    op.execute("DROP EXTENSION IF EXISTS vector")
