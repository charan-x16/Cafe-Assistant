"""Add production catalog and source document schema.

Revision ID: 20260619_0005
Revises: 20260618_0004
Create Date: 2026-06-19 00:00:00.000000

"""
# ruff: noqa: E501
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from cafe_assistant.db.types import EmbeddingVector

revision: str = "20260619_0005"
down_revision: str | None = "20260618_0004"
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
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "source_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("document_type", sa.String(length=100), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("version", sa.String(length=100), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_source_documents_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_source_documents")),
        sa.UniqueConstraint(
            "tenant_id",
            "content_hash",
            name="uq_source_documents_tenant_content_hash",
        ),
    )
    op.create_index(op.f("ix_source_documents_tenant_id"), "source_documents", ["tenant_id"])
    op.create_index(
        op.f("ix_source_documents_document_type"),
        "source_documents",
        ["document_type"],
    )
    op.create_index(
        op.f("ix_source_documents_content_hash"),
        "source_documents",
        ["content_hash"],
    )

    op.create_table(
        "menu_import_batches",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("validation_report", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_menu_import_batches_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_menu_import_batches")),
    )
    op.create_index(op.f("ix_menu_import_batches_tenant_id"), "menu_import_batches", ["tenant_id"])
    op.create_index(op.f("ix_menu_import_batches_status"), "menu_import_batches", ["status"])

    op.create_table(
        "menu_import_batch_documents",
        sa.Column("batch_id", sa.Integer(), nullable=False),
        sa.Column("source_document_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["menu_import_batches.id"],
            name=op.f("fk_menu_import_batch_documents_batch_id_menu_import_batches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_document_id"],
            ["source_documents.id"],
            name=op.f("fk_menu_import_batch_documents_source_document_id_source_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "batch_id",
            "source_document_id",
            name=op.f("pk_menu_import_batch_documents"),
        ),
    )

    op.create_table(
        "menus",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_menus_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_menus")),
        sa.UniqueConstraint("tenant_id", "slug", name="uq_menus_tenant_slug"),
    )
    op.create_index(op.f("ix_menus_tenant_id"), "menus", ["tenant_id"])

    op.create_table(
        "menu_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("menu_id", sa.Integer(), nullable=False),
        sa.Column("import_batch_id", sa.Integer(), nullable=True),
        sa.Column("version", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["import_batch_id"],
            ["menu_import_batches.id"],
            name=op.f("fk_menu_versions_import_batch_id_menu_import_batches"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["menu_id"],
            ["menus.id"],
            name=op.f("fk_menu_versions_menu_id_menus"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_menu_versions")),
        sa.UniqueConstraint("menu_id", "version", name="uq_menu_versions_menu_version"),
    )
    op.create_index(op.f("ix_menu_versions_menu_id"), "menu_versions", ["menu_id"])
    op.create_index(op.f("ix_menu_versions_import_batch_id"), "menu_versions", ["import_batch_id"])
    op.create_index(op.f("ix_menu_versions_status"), "menu_versions", ["status"])

    op.create_table(
        "menu_categories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("menu_version_id", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["menu_version_id"],
            ["menu_versions.id"],
            name=op.f("fk_menu_categories_menu_version_id_menu_versions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["menu_categories.id"],
            name=op.f("fk_menu_categories_parent_id_menu_categories"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_menu_categories")),
        sa.UniqueConstraint(
            "menu_version_id",
            "path",
            name="uq_menu_categories_version_path",
        ),
    )
    op.create_index(op.f("ix_menu_categories_menu_version_id"), "menu_categories", ["menu_version_id"])
    op.create_index(op.f("ix_menu_categories_parent_id"), "menu_categories", ["parent_id"])

    op.create_table(
        "catalog_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("menu_version_id", sa.Integer(), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("source_document_id", sa.Integer(), nullable=True),
        sa.Column("canonical_name", sa.String(length=240), nullable=False),
        sa.Column("display_name", sa.String(length=240), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("source_heading_path", sa.String(length=500), nullable=False),
        sa.Column("tags", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("is_available", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "allergen_data_complete",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "dietary_data_complete",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["category_id"],
            ["menu_categories.id"],
            name=op.f("fk_catalog_items_category_id_menu_categories"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["menu_version_id"],
            ["menu_versions.id"],
            name=op.f("fk_catalog_items_menu_version_id_menu_versions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_document_id"],
            ["source_documents.id"],
            name=op.f("fk_catalog_items_source_document_id_source_documents"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_items")),
        sa.UniqueConstraint(
            "menu_version_id",
            "canonical_name",
            name="uq_catalog_items_version_canonical_name",
        ),
    )
    op.create_index(op.f("ix_catalog_items_menu_version_id"), "catalog_items", ["menu_version_id"])
    op.create_index(op.f("ix_catalog_items_category_id"), "catalog_items", ["category_id"])
    op.create_index(op.f("ix_catalog_items_source_document_id"), "catalog_items", ["source_document_id"])

    op.create_table(
        "catalog_item_variants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("catalog_item_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("serving", sa.String(length=200), nullable=True),
        sa.Column("temperature", sa.String(length=50), nullable=True),
        sa.Column("price_cents", sa.Integer(), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("caffeine_level", sa.String(length=100), nullable=True),
        sa.Column("sweetness_level", sa.String(length=100), nullable=True),
        sa.Column("spice_level", sa.String(length=100), nullable=True),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_available", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_item_id"],
            ["catalog_items.id"],
            name=op.f("fk_catalog_item_variants_catalog_item_id_catalog_items"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_item_variants")),
        sa.UniqueConstraint(
            "catalog_item_id",
            "name",
            name="uq_catalog_item_variants_item_name",
        ),
    )
    op.create_index(op.f("ix_catalog_item_variants_catalog_item_id"), "catalog_item_variants", ["catalog_item_id"])

    op.create_table(
        "catalog_item_nutrition",
        sa.Column("variant_id", sa.Integer(), nullable=False),
        sa.Column("calories", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("protein_grams", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("carbs_grams", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("fat_grams", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("sugar_grams", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("fiber_grams", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("sodium_mg", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("caffeine_mg", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("data_complete", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["variant_id"],
            ["catalog_item_variants.id"],
            name=op.f("fk_catalog_item_nutrition_variant_id_catalog_item_variants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("variant_id", name=op.f("pk_catalog_item_nutrition")),
    )

    op.create_table(
        "catalog_item_ingredients",
        sa.Column("catalog_item_id", sa.Integer(), nullable=False),
        sa.Column("ingredient_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_item_id"],
            ["catalog_items.id"],
            name=op.f("fk_catalog_item_ingredients_catalog_item_id_catalog_items"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["ingredient_id"],
            ["ingredients.id"],
            name=op.f("fk_catalog_item_ingredients_ingredient_id_ingredients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "catalog_item_id",
            "ingredient_id",
            name=op.f("pk_catalog_item_ingredients"),
        ),
    )

    op.create_table(
        "catalog_item_allergen_assertions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("catalog_item_id", sa.Integer(), nullable=False),
        sa.Column("allergen_id", sa.Integer(), nullable=False),
        sa.Column("assertion_type", sa.String(length=50), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("source_document_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["allergen_id"],
            ["allergens.id"],
            name=op.f("fk_catalog_item_allergen_assertions_allergen_id_allergens"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["catalog_item_id"],
            ["catalog_items.id"],
            name=op.f("fk_catalog_item_allergen_assertions_catalog_item_id_catalog_items"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_document_id"],
            ["source_documents.id"],
            name=op.f("fk_catalog_item_allergen_assertions_source_document_id_source_documents"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_item_allergen_assertions")),
        sa.UniqueConstraint(
            "catalog_item_id",
            "allergen_id",
            "assertion_type",
            name="uq_catalog_item_allergen_assertions_item_allergen_type",
        ),
    )
    op.create_index(
        op.f("ix_catalog_item_allergen_assertions_catalog_item_id"),
        "catalog_item_allergen_assertions",
        ["catalog_item_id"],
    )
    op.create_index(
        op.f("ix_catalog_item_allergen_assertions_allergen_id"),
        "catalog_item_allergen_assertions",
        ["allergen_id"],
    )
    op.create_index(
        op.f("ix_catalog_item_allergen_assertions_assertion_type"),
        "catalog_item_allergen_assertions",
        ["assertion_type"],
    )

    op.create_table(
        "catalog_item_dietary_assertions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("catalog_item_id", sa.Integer(), nullable=False),
        sa.Column("dietary_tag_id", sa.Integer(), nullable=False),
        sa.Column("assertion_type", sa.String(length=50), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("source_document_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["catalog_item_id"],
            ["catalog_items.id"],
            name=op.f("fk_catalog_item_dietary_assertions_catalog_item_id_catalog_items"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dietary_tag_id"],
            ["dietary_tags.id"],
            name=op.f("fk_catalog_item_dietary_assertions_dietary_tag_id_dietary_tags"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_document_id"],
            ["source_documents.id"],
            name=op.f("fk_catalog_item_dietary_assertions_source_document_id_source_documents"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_item_dietary_assertions")),
        sa.UniqueConstraint(
            "catalog_item_id",
            "dietary_tag_id",
            "assertion_type",
            name="uq_catalog_item_dietary_assertions_item_tag_type",
        ),
    )
    op.create_index(
        op.f("ix_catalog_item_dietary_assertions_catalog_item_id"),
        "catalog_item_dietary_assertions",
        ["catalog_item_id"],
    )
    op.create_index(
        op.f("ix_catalog_item_dietary_assertions_dietary_tag_id"),
        "catalog_item_dietary_assertions",
        ["dietary_tag_id"],
    )
    op.create_index(
        op.f("ix_catalog_item_dietary_assertions_assertion_type"),
        "catalog_item_dietary_assertions",
        ["assertion_type"],
    )

    op.create_table(
        "modifier_groups",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("menu_version_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("selection_type", sa.String(length=50), nullable=False),
        sa.Column("min_select", sa.Integer(), nullable=False),
        sa.Column("max_select", sa.Integer(), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["menu_version_id"],
            ["menu_versions.id"],
            name=op.f("fk_modifier_groups_menu_version_id_menu_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_modifier_groups")),
        sa.UniqueConstraint("menu_version_id", "name", name="uq_modifier_groups_version_name"),
    )
    op.create_index(op.f("ix_modifier_groups_menu_version_id"), "modifier_groups", ["menu_version_id"])

    op.create_table(
        "modifier_options",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("group_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("price_delta_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column(
            "allergen_data_complete",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "dietary_data_complete",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("is_available", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("metadata_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["modifier_groups.id"],
            name=op.f("fk_modifier_options_group_id_modifier_groups"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_modifier_options")),
        sa.UniqueConstraint("group_id", "name", name="uq_modifier_options_group_name"),
    )
    op.create_index(op.f("ix_modifier_options_group_id"), "modifier_options", ["group_id"])

    op.create_table(
        "catalog_item_modifier_groups",
        sa.Column("catalog_item_id", sa.Integer(), nullable=False),
        sa.Column("modifier_group_id", sa.Integer(), nullable=False),
        sa.Column("is_required", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_item_id"],
            ["catalog_items.id"],
            name=op.f("fk_catalog_item_modifier_groups_catalog_item_id_catalog_items"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["modifier_group_id"],
            ["modifier_groups.id"],
            name=op.f("fk_catalog_item_modifier_groups_modifier_group_id_modifier_groups"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "catalog_item_id",
            "modifier_group_id",
            name=op.f("pk_catalog_item_modifier_groups"),
        ),
    )

    op.create_table(
        "modifier_option_allergen_assertions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("modifier_option_id", sa.Integer(), nullable=False),
        sa.Column("allergen_id", sa.Integer(), nullable=False),
        sa.Column("assertion_type", sa.String(length=50), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["allergen_id"],
            ["allergens.id"],
            name=op.f("fk_modifier_option_allergen_assertions_allergen_id_allergens"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["modifier_option_id"],
            ["modifier_options.id"],
            name=op.f("fk_modifier_option_allergen_assertions_modifier_option_id_modifier_options"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_modifier_option_allergen_assertions")),
        sa.UniqueConstraint(
            "modifier_option_id",
            "allergen_id",
            "assertion_type",
            name="uq_modifier_option_allergen_assertions_option_allergen_type",
        ),
    )
    op.create_index(
        op.f("ix_modifier_option_allergen_assertions_modifier_option_id"),
        "modifier_option_allergen_assertions",
        ["modifier_option_id"],
    )
    op.create_index(
        op.f("ix_modifier_option_allergen_assertions_allergen_id"),
        "modifier_option_allergen_assertions",
        ["allergen_id"],
    )
    op.create_index(
        op.f("ix_modifier_option_allergen_assertions_assertion_type"),
        "modifier_option_allergen_assertions",
        ["assertion_type"],
    )

    op.create_table(
        "modifier_option_dietary_assertions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("modifier_option_id", sa.Integer(), nullable=False),
        sa.Column("dietary_tag_id", sa.Integer(), nullable=False),
        sa.Column("assertion_type", sa.String(length=50), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dietary_tag_id"],
            ["dietary_tags.id"],
            name=op.f("fk_modifier_option_dietary_assertions_dietary_tag_id_dietary_tags"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["modifier_option_id"],
            ["modifier_options.id"],
            name=op.f("fk_modifier_option_dietary_assertions_modifier_option_id_modifier_options"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_modifier_option_dietary_assertions")),
        sa.UniqueConstraint(
            "modifier_option_id",
            "dietary_tag_id",
            "assertion_type",
            name="uq_modifier_option_dietary_assertions_option_tag_type",
        ),
    )
    op.create_index(
        op.f("ix_modifier_option_dietary_assertions_modifier_option_id"),
        "modifier_option_dietary_assertions",
        ["modifier_option_id"],
    )
    op.create_index(
        op.f("ix_modifier_option_dietary_assertions_dietary_tag_id"),
        "modifier_option_dietary_assertions",
        ["dietary_tag_id"],
    )
    op.create_index(
        op.f("ix_modifier_option_dietary_assertions_assertion_type"),
        "modifier_option_dietary_assertions",
        ["assertion_type"],
    )

    op.create_table(
        "catalog_item_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("variant_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedded_text_hash", sa.String(length=128), nullable=False),
        sa.Column("embedded_text", sa.Text(), nullable=False),
        sa.Column("embedding", EmbeddingVector(8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["variant_id"],
            ["catalog_item_variants.id"],
            name=op.f("fk_catalog_item_embeddings_variant_id_catalog_item_variants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_item_embeddings")),
        sa.UniqueConstraint(
            "variant_id",
            "provider",
            "model_name",
            "embedded_text_hash",
            name="uq_catalog_item_embeddings_variant_provider_model_hash",
        ),
    )
    op.create_index(op.f("ix_catalog_item_embeddings_variant_id"), "catalog_item_embeddings", ["variant_id"])
    op.create_index(
        op.f("ix_catalog_item_embeddings_embedded_text_hash"),
        "catalog_item_embeddings",
        ["embedded_text_hash"],
    )
    op.execute(
        """
        CREATE INDEX ix_catalog_item_embeddings_embedding_hnsw
        ON catalog_item_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )

    op.create_table(
        "policy_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("source_document_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("version", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_document_id"],
            ["source_documents.id"],
            name=op.f("fk_policy_documents_source_document_id_source_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_policy_documents_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_documents")),
        sa.UniqueConstraint(
            "tenant_id",
            "source_document_id",
            name="uq_policy_documents_tenant_source",
        ),
    )
    op.create_index(op.f("ix_policy_documents_tenant_id"), "policy_documents", ["tenant_id"])
    op.create_index(
        op.f("ix_policy_documents_source_document_id"),
        "policy_documents",
        ["source_document_id"],
    )

    op.create_table(
        "policy_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("policy_document_id", sa.Integer(), nullable=False),
        sa.Column("heading_path", sa.String(length=500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["policy_document_id"],
            ["policy_documents.id"],
            name=op.f("fk_policy_chunks_policy_document_id_policy_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_chunks")),
        sa.UniqueConstraint(
            "policy_document_id",
            "chunk_index",
            name="uq_policy_chunks_document_chunk_index",
        ),
    )
    op.create_index(op.f("ix_policy_chunks_policy_document_id"), "policy_chunks", ["policy_document_id"])
    op.create_index(op.f("ix_policy_chunks_content_hash"), "policy_chunks", ["content_hash"])

    op.create_table(
        "policy_chunk_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("policy_chunk_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedded_text_hash", sa.String(length=128), nullable=False),
        sa.Column("embedded_text", sa.Text(), nullable=False),
        sa.Column("embedding", EmbeddingVector(8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["policy_chunk_id"],
            ["policy_chunks.id"],
            name=op.f("fk_policy_chunk_embeddings_policy_chunk_id_policy_chunks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_chunk_embeddings")),
        sa.UniqueConstraint(
            "policy_chunk_id",
            "provider",
            "model_name",
            "embedded_text_hash",
            name="uq_policy_chunk_embeddings_chunk_provider_model_hash",
        ),
    )
    op.create_index(op.f("ix_policy_chunk_embeddings_policy_chunk_id"), "policy_chunk_embeddings", ["policy_chunk_id"])
    op.create_index(
        op.f("ix_policy_chunk_embeddings_embedded_text_hash"),
        "policy_chunk_embeddings",
        ["embedded_text_hash"],
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
    op.execute("DROP INDEX IF EXISTS ix_policy_chunk_embeddings_embedding_hnsw")
    op.drop_index(op.f("ix_policy_chunk_embeddings_embedded_text_hash"), table_name="policy_chunk_embeddings")
    op.drop_index(op.f("ix_policy_chunk_embeddings_policy_chunk_id"), table_name="policy_chunk_embeddings")
    op.drop_table("policy_chunk_embeddings")
    op.drop_index(op.f("ix_policy_chunks_content_hash"), table_name="policy_chunks")
    op.drop_index(op.f("ix_policy_chunks_policy_document_id"), table_name="policy_chunks")
    op.drop_table("policy_chunks")
    op.drop_index(op.f("ix_policy_documents_source_document_id"), table_name="policy_documents")
    op.drop_index(op.f("ix_policy_documents_tenant_id"), table_name="policy_documents")
    op.drop_table("policy_documents")
    op.execute("DROP INDEX IF EXISTS ix_catalog_item_embeddings_embedding_hnsw")
    op.drop_index(op.f("ix_catalog_item_embeddings_embedded_text_hash"), table_name="catalog_item_embeddings")
    op.drop_index(op.f("ix_catalog_item_embeddings_variant_id"), table_name="catalog_item_embeddings")
    op.drop_table("catalog_item_embeddings")
    op.drop_index(
        op.f("ix_modifier_option_dietary_assertions_assertion_type"),
        table_name="modifier_option_dietary_assertions",
    )
    op.drop_index(
        op.f("ix_modifier_option_dietary_assertions_dietary_tag_id"),
        table_name="modifier_option_dietary_assertions",
    )
    op.drop_index(
        op.f("ix_modifier_option_dietary_assertions_modifier_option_id"),
        table_name="modifier_option_dietary_assertions",
    )
    op.drop_table("modifier_option_dietary_assertions")
    op.drop_index(
        op.f("ix_modifier_option_allergen_assertions_assertion_type"),
        table_name="modifier_option_allergen_assertions",
    )
    op.drop_index(
        op.f("ix_modifier_option_allergen_assertions_allergen_id"),
        table_name="modifier_option_allergen_assertions",
    )
    op.drop_index(
        op.f("ix_modifier_option_allergen_assertions_modifier_option_id"),
        table_name="modifier_option_allergen_assertions",
    )
    op.drop_table("modifier_option_allergen_assertions")
    op.drop_table("catalog_item_modifier_groups")
    op.drop_index(op.f("ix_modifier_options_group_id"), table_name="modifier_options")
    op.drop_table("modifier_options")
    op.drop_index(op.f("ix_modifier_groups_menu_version_id"), table_name="modifier_groups")
    op.drop_table("modifier_groups")
    op.drop_index(
        op.f("ix_catalog_item_dietary_assertions_assertion_type"),
        table_name="catalog_item_dietary_assertions",
    )
    op.drop_index(
        op.f("ix_catalog_item_dietary_assertions_dietary_tag_id"),
        table_name="catalog_item_dietary_assertions",
    )
    op.drop_index(
        op.f("ix_catalog_item_dietary_assertions_catalog_item_id"),
        table_name="catalog_item_dietary_assertions",
    )
    op.drop_table("catalog_item_dietary_assertions")
    op.drop_index(
        op.f("ix_catalog_item_allergen_assertions_assertion_type"),
        table_name="catalog_item_allergen_assertions",
    )
    op.drop_index(
        op.f("ix_catalog_item_allergen_assertions_allergen_id"),
        table_name="catalog_item_allergen_assertions",
    )
    op.drop_index(
        op.f("ix_catalog_item_allergen_assertions_catalog_item_id"),
        table_name="catalog_item_allergen_assertions",
    )
    op.drop_table("catalog_item_allergen_assertions")
    op.drop_table("catalog_item_ingredients")
    op.drop_table("catalog_item_nutrition")
    op.drop_index(op.f("ix_catalog_item_variants_catalog_item_id"), table_name="catalog_item_variants")
    op.drop_table("catalog_item_variants")
    op.drop_index(op.f("ix_catalog_items_source_document_id"), table_name="catalog_items")
    op.drop_index(op.f("ix_catalog_items_category_id"), table_name="catalog_items")
    op.drop_index(op.f("ix_catalog_items_menu_version_id"), table_name="catalog_items")
    op.drop_table("catalog_items")
    op.drop_index(op.f("ix_menu_categories_parent_id"), table_name="menu_categories")
    op.drop_index(op.f("ix_menu_categories_menu_version_id"), table_name="menu_categories")
    op.drop_table("menu_categories")
    op.drop_index(op.f("ix_menu_versions_status"), table_name="menu_versions")
    op.drop_index(op.f("ix_menu_versions_import_batch_id"), table_name="menu_versions")
    op.drop_index(op.f("ix_menu_versions_menu_id"), table_name="menu_versions")
    op.drop_table("menu_versions")
    op.drop_index(op.f("ix_menus_tenant_id"), table_name="menus")
    op.drop_table("menus")
    op.drop_table("menu_import_batch_documents")
    op.drop_index(op.f("ix_menu_import_batches_status"), table_name="menu_import_batches")
    op.drop_index(op.f("ix_menu_import_batches_tenant_id"), table_name="menu_import_batches")
    op.drop_table("menu_import_batches")
    op.drop_index(op.f("ix_source_documents_content_hash"), table_name="source_documents")
    op.drop_index(op.f("ix_source_documents_document_type"), table_name="source_documents")
    op.drop_index(op.f("ix_source_documents_tenant_id"), table_name="source_documents")
    op.drop_table("source_documents")
