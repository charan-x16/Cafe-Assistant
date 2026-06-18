from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
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


class Tenant(Base):
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


class Location(Base):
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
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(8), nullable=True)
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
    __tablename__ = "allergens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    ingredients: Mapped[list[Ingredient]] = relationship(
        secondary=ingredient_allergens,
        back_populates="allergens",
    )


class DietaryTag(Base):
    __tablename__ = "dietary_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    menu_items: Mapped[list[MenuItem]] = relationship(
        secondary=item_dietary_tags,
        back_populates="dietary_tags",
    )


class Customer(Base):
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
    __tablename__ = "consents"

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
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tenant: Mapped[Tenant] = relationship(back_populates="device_tokens")
    customer: Mapped[Customer] = relationship(back_populates="device_tokens")


class AuditEvent(Base):
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
