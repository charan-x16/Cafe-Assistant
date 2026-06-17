from __future__ import annotations

from decimal import Decimal

from sqlalchemy import Boolean, Column, ForeignKey, Integer, Numeric, String, Table, text
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
