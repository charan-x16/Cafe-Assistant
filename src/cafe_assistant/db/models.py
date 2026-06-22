"""SQLAlchemy ORM model definitions for the cafe assistant data layer.
Includes legacy menu tables, production catalog tables, identity and memory records, consent
records, embeddings, policy chunks, and audit events.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cafe_assistant.db.base import Base
from cafe_assistant.db.types import EmbeddingVector

item_ingredients = Table(
    "item_ingredients",
    Base.metadata,
    Column(
        "item_id",
        ForeignKey("menu_items.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "ingredient_id",
        ForeignKey("ingredients.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

ingredient_allergens = Table(
    "ingredient_allergens",
    Base.metadata,
    Column(
        "ingredient_id",
        ForeignKey("ingredients.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "allergen_id",
        ForeignKey("allergens.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

item_dietary_tags = Table(
    "item_dietary_tags",
    Base.metadata,
    Column(
        "item_id",
        ForeignKey("menu_items.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "dietary_tag_id",
        ForeignKey("dietary_tags.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

menu_import_batch_documents = Table(
    "menu_import_batch_documents",
    Base.metadata,
    Column(
        "batch_id",
        ForeignKey("menu_import_batches.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "source_document_id",
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

catalog_item_ingredients = Table(
    "catalog_item_ingredients",
    Base.metadata,
    Column(
        "catalog_item_id",
        ForeignKey("catalog_items.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "ingredient_id",
        ForeignKey("ingredients.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

catalog_item_modifier_groups = Table(
    "catalog_item_modifier_groups",
    Base.metadata,
    Column(
        "catalog_item_id",
        ForeignKey("catalog_items.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "modifier_group_id",
        ForeignKey("modifier_groups.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "is_required",
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
    ),
)


class Tenant(Base):
    """SQLAlchemy model for tenant records."""
    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)

    locations: Mapped[list[Location]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    menu_items: Mapped[list[MenuItem]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    customers: Mapped[list[Customer]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    device_tokens: Mapped[list[CustomerDeviceToken]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    audit_events: Mapped[list[AuditEvent]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    source_documents: Mapped[list[SourceDocument]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    menu_import_batches: Mapped[list[MenuImportBatch]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    menus: Mapped[list[Menu]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    policy_documents: Mapped[list[PolicyDocument]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )


class Location(Base):
    """SQLAlchemy model for location records."""
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    tenant: Mapped[Tenant] = relationship(back_populates="locations")


class MenuItem(Base):
    """SQLAlchemy model for menu item records."""
    __tablename__ = "menu_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    sugar_grams: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    carbs_grams: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(384), nullable=True)
    is_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    allergen_data_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
    )

    tenant: Mapped[Tenant] = relationship(back_populates="menu_items")
    ingredients: Mapped[list[Ingredient]] = relationship(
        secondary=item_ingredients,
        back_populates="menu_items",
    )
    dietary_tags: Mapped[list[DietaryTag]] = relationship(
        secondary=item_dietary_tags,
        back_populates="menu_items",
    )


class Ingredient(Base):
    """SQLAlchemy model for ingredient records."""
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)

    menu_items: Mapped[list[MenuItem]] = relationship(
        secondary=item_ingredients,
        back_populates="ingredients",
    )
    allergens: Mapped[list[Allergen]] = relationship(
        secondary=ingredient_allergens,
        back_populates="ingredients",
    )


class Allergen(Base):
    """SQLAlchemy model for allergen records."""
    __tablename__ = "allergens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    ingredients: Mapped[list[Ingredient]] = relationship(
        secondary=ingredient_allergens,
        back_populates="allergens",
    )


class DietaryTag(Base):
    """SQLAlchemy model for dietary tag records."""
    __tablename__ = "dietary_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    menu_items: Mapped[list[MenuItem]] = relationship(
        secondary=item_dietary_tags,
        back_populates="dietary_tags",
    )


class SourceDocument(Base):
    """SQLAlchemy model for source document records."""
    __tablename__ = "source_documents"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "content_hash",
            name="uq_source_documents_tenant_content_hash",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False, default="unversioned")
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="source_documents")
    import_batches: Mapped[list[MenuImportBatch]] = relationship(
        secondary=menu_import_batch_documents,
        back_populates="source_documents",
    )


class MenuImportBatch(Base):
    """SQLAlchemy model for menu import batch records."""
    __tablename__ = "menu_import_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="staged", index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_report: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tenant: Mapped[Tenant] = relationship(back_populates="menu_import_batches")
    source_documents: Mapped[list[SourceDocument]] = relationship(
        secondary=menu_import_batch_documents,
        back_populates="import_batches",
    )
    menu_versions: Mapped[list[MenuVersion]] = relationship(back_populates="import_batch")


class Menu(Base):
    """SQLAlchemy model for menu records."""
    __tablename__ = "menus"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_menus_tenant_slug"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="menus")
    versions: Mapped[list[MenuVersion]] = relationship(
        back_populates="menu",
        cascade="all, delete-orphan",
    )


class MenuVersion(Base):
    """SQLAlchemy model for menu version records."""
    __tablename__ = "menu_versions"
    __table_args__ = (
        UniqueConstraint("menu_id", "version", name="uq_menu_versions_menu_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    menu_id: Mapped[int] = mapped_column(
        ForeignKey("menus.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    import_batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("menu_import_batches.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="staged", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    menu: Mapped[Menu] = relationship(back_populates="versions")
    import_batch: Mapped[MenuImportBatch | None] = relationship(back_populates="menu_versions")
    categories: Mapped[list[MenuCategory]] = relationship(
        back_populates="menu_version",
        cascade="all, delete-orphan",
    )
    items: Mapped[list[CatalogItem]] = relationship(
        back_populates="menu_version",
        cascade="all, delete-orphan",
    )
    modifier_groups: Mapped[list[ModifierGroup]] = relationship(
        back_populates="menu_version",
        cascade="all, delete-orphan",
    )


class MenuCategory(Base):
    """SQLAlchemy model for menu category records."""
    __tablename__ = "menu_categories"
    __table_args__ = (
        UniqueConstraint("menu_version_id", "path", name="uq_menu_categories_version_path"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    menu_version_id: Mapped[int] = mapped_column(
        ForeignKey("menu_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("menu_categories.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    menu_version: Mapped[MenuVersion] = relationship(back_populates="categories")
    parent: Mapped[MenuCategory | None] = relationship(
        remote_side="MenuCategory.id",
        back_populates="children",
    )
    children: Mapped[list[MenuCategory]] = relationship(back_populates="parent")
    items: Mapped[list[CatalogItem]] = relationship(back_populates="category")


class CatalogItem(Base):
    """SQLAlchemy model for catalog item records."""
    __tablename__ = "catalog_items"
    __table_args__ = (
        UniqueConstraint(
            "menu_version_id",
            "canonical_name",
            name="uq_catalog_items_version_canonical_name",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    menu_version_id: Mapped[int] = mapped_column(
        ForeignKey("menu_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("menu_categories.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    canonical_name: Mapped[str] = mapped_column(String(240), nullable=False)
    display_name: Mapped[str] = mapped_column(String(240), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_heading_path: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    tags: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    is_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    allergen_data_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
    )
    dietary_data_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    menu_version: Mapped[MenuVersion] = relationship(back_populates="items")
    category: Mapped[MenuCategory | None] = relationship(back_populates="items")
    source_document: Mapped[SourceDocument | None] = relationship()
    ingredients: Mapped[list[Ingredient]] = relationship(secondary=catalog_item_ingredients)
    variants: Mapped[list[CatalogItemVariant]] = relationship(
        back_populates="catalog_item",
        cascade="all, delete-orphan",
    )
    allergen_assertions: Mapped[list[CatalogItemAllergenAssertion]] = relationship(
        back_populates="catalog_item",
        cascade="all, delete-orphan",
    )
    dietary_assertions: Mapped[list[CatalogItemDietaryAssertion]] = relationship(
        back_populates="catalog_item",
        cascade="all, delete-orphan",
    )
    modifier_groups: Mapped[list[ModifierGroup]] = relationship(
        secondary=catalog_item_modifier_groups,
        back_populates="catalog_items",
    )


class CatalogItemVariant(Base):
    """SQLAlchemy model for catalog item variant records."""
    __tablename__ = "catalog_item_variants"
    __table_args__ = (
        UniqueConstraint("catalog_item_id", "name", name="uq_catalog_item_variants_item_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    catalog_item_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="Default")
    serving: Mapped[str | None] = mapped_column(String(200), nullable=True)
    temperature: Mapped[str | None] = mapped_column(String(50), nullable=True)
    price_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    caffeine_level: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sweetness_level: Mapped[str | None] = mapped_column(String(100), nullable=True)
    spice_level: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    is_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    catalog_item: Mapped[CatalogItem] = relationship(back_populates="variants")
    nutrition: Mapped[CatalogItemNutrition | None] = relationship(
        back_populates="variant",
        cascade="all, delete-orphan",
        uselist=False,
    )
    embeddings: Mapped[list[CatalogItemEmbedding]] = relationship(
        back_populates="variant",
        cascade="all, delete-orphan",
    )


class CatalogItemNutrition(Base):
    """SQLAlchemy model for catalog item nutrition records."""
    __tablename__ = "catalog_item_nutrition"

    variant_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_item_variants.id", ondelete="CASCADE"),
        primary_key=True,
    )
    calories: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    protein_grams: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    carbs_grams: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    fat_grams: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    sugar_grams: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    fiber_grams: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    sodium_mg: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    caffeine_mg: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    data_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    variant: Mapped[CatalogItemVariant] = relationship(back_populates="nutrition")


class CatalogItemAllergenAssertion(Base):
    """SQLAlchemy model for catalog item allergen assertion records."""
    __tablename__ = "catalog_item_allergen_assertions"
    __table_args__ = (
        UniqueConstraint(
            "catalog_item_id",
            "allergen_id",
            "assertion_type",
            name="uq_catalog_item_allergen_assertions_item_allergen_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    catalog_item_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    allergen_id: Mapped[int] = mapped_column(
        ForeignKey("allergens.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assertion_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    catalog_item: Mapped[CatalogItem] = relationship(back_populates="allergen_assertions")
    allergen: Mapped[Allergen] = relationship()
    source_document: Mapped[SourceDocument | None] = relationship()


class CatalogItemDietaryAssertion(Base):
    """SQLAlchemy model for catalog item dietary assertion records."""
    __tablename__ = "catalog_item_dietary_assertions"
    __table_args__ = (
        UniqueConstraint(
            "catalog_item_id",
            "dietary_tag_id",
            "assertion_type",
            name="uq_catalog_item_dietary_assertions_item_tag_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    catalog_item_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dietary_tag_id: Mapped[int] = mapped_column(
        ForeignKey("dietary_tags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assertion_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    catalog_item: Mapped[CatalogItem] = relationship(back_populates="dietary_assertions")
    dietary_tag: Mapped[DietaryTag] = relationship()
    source_document: Mapped[SourceDocument | None] = relationship()


class ModifierGroup(Base):
    """SQLAlchemy model for modifier group records."""
    __tablename__ = "modifier_groups"
    __table_args__ = (
        UniqueConstraint("menu_version_id", "name", name="uq_modifier_groups_version_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    menu_version_id: Mapped[int] = mapped_column(
        ForeignKey("menu_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    selection_type: Mapped[str] = mapped_column(String(50), nullable=False, default="optional")
    min_select: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_select: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    menu_version: Mapped[MenuVersion] = relationship(back_populates="modifier_groups")
    options: Mapped[list[ModifierOption]] = relationship(
        back_populates="group",
        cascade="all, delete-orphan",
    )
    catalog_items: Mapped[list[CatalogItem]] = relationship(
        secondary=catalog_item_modifier_groups,
        back_populates="modifier_groups",
    )


class ModifierOption(Base):
    """SQLAlchemy model for modifier option records."""
    __tablename__ = "modifier_options"
    __table_args__ = (
        UniqueConstraint("group_id", "name", name="uq_modifier_options_group_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("modifier_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    price_delta_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    allergen_data_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
    )
    dietary_data_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
    )
    is_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    metadata_json: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    group: Mapped[ModifierGroup] = relationship(back_populates="options")
    allergen_assertions: Mapped[list[ModifierOptionAllergenAssertion]] = relationship(
        back_populates="modifier_option",
        cascade="all, delete-orphan",
    )
    dietary_assertions: Mapped[list[ModifierOptionDietaryAssertion]] = relationship(
        back_populates="modifier_option",
        cascade="all, delete-orphan",
    )


class ModifierOptionAllergenAssertion(Base):
    """SQLAlchemy model for modifier option allergen assertion records."""
    __tablename__ = "modifier_option_allergen_assertions"
    __table_args__ = (
        UniqueConstraint(
            "modifier_option_id",
            "allergen_id",
            "assertion_type",
            name="uq_modifier_option_allergen_assertions_option_allergen_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    modifier_option_id: Mapped[int] = mapped_column(
        ForeignKey("modifier_options.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    allergen_id: Mapped[int] = mapped_column(
        ForeignKey("allergens.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assertion_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)

    modifier_option: Mapped[ModifierOption] = relationship(back_populates="allergen_assertions")
    allergen: Mapped[Allergen] = relationship()


class ModifierOptionDietaryAssertion(Base):
    """SQLAlchemy model for modifier option dietary assertion records."""
    __tablename__ = "modifier_option_dietary_assertions"
    __table_args__ = (
        UniqueConstraint(
            "modifier_option_id",
            "dietary_tag_id",
            "assertion_type",
            name="uq_modifier_option_dietary_assertions_option_tag_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    modifier_option_id: Mapped[int] = mapped_column(
        ForeignKey("modifier_options.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dietary_tag_id: Mapped[int] = mapped_column(
        ForeignKey("dietary_tags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assertion_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)

    modifier_option: Mapped[ModifierOption] = relationship(back_populates="dietary_assertions")
    dietary_tag: Mapped[DietaryTag] = relationship()


class CatalogItemEmbedding(Base):
    """SQLAlchemy model for catalog item embedding records."""
    __tablename__ = "catalog_item_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "variant_id",
            "provider",
            "model_name",
            "embedded_text_hash",
            name="uq_catalog_item_embeddings_variant_provider_model_hash",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    variant_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_item_variants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedded_text_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    embedded_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    variant: Mapped[CatalogItemVariant] = relationship(back_populates="embeddings")


class PolicyDocument(Base):
    """SQLAlchemy model for policy document records."""
    __tablename__ = "policy_documents"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_document_id",
            name="uq_policy_documents_tenant_source",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False, default="unversioned")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="policy_documents")
    source_document: Mapped[SourceDocument] = relationship()
    chunks: Mapped[list[PolicyChunk]] = relationship(
        back_populates="policy_document",
        cascade="all, delete-orphan",
    )


class PolicyChunk(Base):
    """SQLAlchemy model for policy chunk records."""
    __tablename__ = "policy_chunks"
    __table_args__ = (
        UniqueConstraint(
            "policy_document_id",
            "chunk_index",
            name="uq_policy_chunks_document_chunk_index",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_document_id: Mapped[int] = mapped_column(
        ForeignKey("policy_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    heading_path: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    policy_document: Mapped[PolicyDocument] = relationship(back_populates="chunks")
    embeddings: Mapped[list[PolicyChunkEmbedding]] = relationship(
        back_populates="policy_chunk",
        cascade="all, delete-orphan",
    )


class PolicyChunkEmbedding(Base):
    """SQLAlchemy model for policy chunk embedding records."""
    __tablename__ = "policy_chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "policy_chunk_id",
            "provider",
            "model_name",
            "embedded_text_hash",
            name="uq_policy_chunk_embeddings_chunk_provider_model_hash",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_chunk_id: Mapped[int] = mapped_column(
        ForeignKey("policy_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedded_text_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    embedded_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    policy_chunk: Mapped[PolicyChunk] = relationship(back_populates="embeddings")


class Customer(Base):
    """SQLAlchemy model for customer records."""
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "phone_hash",
            name="uq_customers_tenant_phone_hash",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    phone_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="customers")
    profile: Mapped[CustomerProfile | None] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        uselist=False,
    )
    events: Mapped[list[EpisodicEvent]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
    )
    consents: Mapped[list[Consent]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
    )
    device_tokens: Mapped[list[CustomerDeviceToken]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
    )


class CustomerProfile(Base):
    """SQLAlchemy model for customer profile records."""
    __tablename__ = "customer_profile"

    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        primary_key=True,
    )
    preferences: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    dietary_facts: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    customer: Mapped[Customer] = relationship(back_populates="profile")


class EpisodicEvent(Base):
    """SQLAlchemy model for episodic event records."""
    __tablename__ = "episodic_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    customer: Mapped[Customer] = relationship(back_populates="events")


class Consent(Base):
    """SQLAlchemy model for consent records."""
    __tablename__ = "consents"
    __table_args__ = (
        Index(
            "uq_consents_active_customer_scope",
            "customer_id",
            "scope",
            unique=True,
            postgresql_where=text("revoked_at IS NULL"),
            sqlite_where=text("revoked_at IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scope: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    customer: Mapped[Customer] = relationship(back_populates="consents")


class CustomerDeviceToken(Base):
    """SQLAlchemy model for customer device token records."""
    __tablename__ = "customer_device_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tenant: Mapped[Tenant] = relationship(back_populates="device_tokens")
    customer: Mapped[Customer] = relationship(back_populates="device_tokens")


class AuditEvent(Base):
    """SQLAlchemy model for audit event records."""
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    actor: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    trace_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    payload_redacted: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="audit_events")
