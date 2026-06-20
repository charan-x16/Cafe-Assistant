"""Tests for legacy menu.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.models import Allergen, DietaryTag, Ingredient, Location, MenuItem, Tenant


class MenuItemSeed(TypedDict):
    """Container for menu item seed behavior and data."""
    name: str
    description: str
    price_cents: int
    category: str
    sugar_grams: str | None
    carbs_grams: str | None
    is_available: bool
    allergen_data_complete: bool
    ingredients: list[str]
    dietary_tags: list[str]


TENANT_NAME = "Acorn & Steam Cafe"
LOCATION_NAME = "Main Street Cafe"

ALLERGENS: tuple[tuple[str, str], ...] = (
    ("PEANUT", "Peanut"),
    ("TREE_NUT", "Tree nut"),
    ("DAIRY", "Dairy"),
    ("GLUTEN", "Gluten"),
    ("SOY", "Soy"),
    ("EGG", "Egg"),
)

DIETARY_TAGS: tuple[tuple[str, str], ...] = (
    ("VEGAN", "Vegan"),
    ("VEGETARIAN", "Vegetarian"),
    ("GLUTEN_FREE", "Gluten free"),
)

INGREDIENT_ALLERGEN_CODES: dict[str, tuple[str, ...]] = {
    "almond milk": ("TREE_NUT",),
    "almonds": ("TREE_NUT",),
    "brioche": ("GLUTEN", "DAIRY", "EGG"),
    "butter": ("DAIRY",),
    "cheddar": ("DAIRY",),
    "chocolate chips": ("DAIRY", "SOY"),
    "cookie dough": ("GLUTEN", "DAIRY", "EGG"),
    "croissant dough": ("GLUTEN", "DAIRY"),
    "egg": ("EGG",),
    "feta": ("DAIRY",),
    "honey oat bread": ("GLUTEN",),
    "milk chocolate sauce": ("DAIRY", "SOY"),
    "mozzarella": ("DAIRY",),
    "oat milk": ("GLUTEN",),
    "peanut butter": ("PEANUT",),
    "pesto": ("TREE_NUT", "DAIRY"),
    "spinach tortilla": ("GLUTEN",),
    "sourdough": ("GLUTEN",),
    "vanilla syrup": (),
    "wheat flour": ("GLUTEN",),
    "whole milk": ("DAIRY",),
}

MENU_ITEMS: tuple[MenuItemSeed, ...] = (
    {
        "name": "Espresso",
        "description": "Double shot of house espresso.",
        "price_cents": 325,
        "category": "Coffee",
        "sugar_grams": "0.0",
        "carbs_grams": "1.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["espresso"],
        "dietary_tags": ["VEGAN", "VEGETARIAN", "GLUTEN_FREE"],
    },
    {
        "name": "Americano",
        "description": "House espresso lengthened with hot filtered water.",
        "price_cents": 375,
        "category": "Coffee",
        "sugar_grams": "0.0",
        "carbs_grams": "1.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["espresso", "filtered water"],
        "dietary_tags": ["VEGAN", "VEGETARIAN", "GLUTEN_FREE"],
    },
    {
        "name": "Cappuccino",
        "description": "Espresso with steamed whole milk and dense foam.",
        "price_cents": 475,
        "category": "Coffee",
        "sugar_grams": "9.0",
        "carbs_grams": "12.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["espresso", "whole milk"],
        "dietary_tags": ["VEGETARIAN", "GLUTEN_FREE"],
    },
    {
        "name": "Vanilla Oat Latte",
        "description": "Espresso, oat milk, and vanilla syrup.",
        "price_cents": 575,
        "category": "Coffee",
        "sugar_grams": "24.0",
        "carbs_grams": "34.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["espresso", "oat milk", "vanilla syrup"],
        "dietary_tags": ["VEGAN", "VEGETARIAN"],
    },
    {
        "name": "Mocha",
        "description": "Espresso with whole milk and milk chocolate sauce.",
        "price_cents": 595,
        "category": "Coffee",
        "sugar_grams": "31.0",
        "carbs_grams": "42.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["espresso", "whole milk", "milk chocolate sauce"],
        "dietary_tags": ["VEGETARIAN"],
    },
    {
        "name": "Cold Brew",
        "description": "Slow-steeped coffee served over ice.",
        "price_cents": 450,
        "category": "Coffee",
        "sugar_grams": "0.0",
        "carbs_grams": "2.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["cold brew coffee"],
        "dietary_tags": ["VEGAN", "VEGETARIAN", "GLUTEN_FREE"],
    },
    {
        "name": "Chai Latte",
        "description": "Black tea and warming spices with steamed whole milk.",
        "price_cents": 525,
        "category": "Tea",
        "sugar_grams": "28.0",
        "carbs_grams": "36.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["chai concentrate", "whole milk"],
        "dietary_tags": ["VEGETARIAN", "GLUTEN_FREE"],
    },
    {
        "name": "Matcha Almond Latte",
        "description": "Ceremonial-style matcha whisked with almond milk.",
        "price_cents": 575,
        "category": "Tea",
        "sugar_grams": "14.0",
        "carbs_grams": "18.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["matcha", "almond milk"],
        "dietary_tags": ["VEGAN", "VEGETARIAN", "GLUTEN_FREE"],
    },
    {
        "name": "Earl Grey Tea",
        "description": "Bergamot-scented black tea served hot.",
        "price_cents": 325,
        "category": "Tea",
        "sugar_grams": "0.0",
        "carbs_grams": "0.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["earl grey tea"],
        "dietary_tags": ["VEGAN", "VEGETARIAN", "GLUTEN_FREE"],
    },
    {
        "name": "Blueberry Muffin",
        "description": "Tender muffin with blueberries and cinnamon crumb.",
        "price_cents": 425,
        "category": "Pastry",
        "sugar_grams": "27.0",
        "carbs_grams": "52.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["wheat flour", "butter", "egg", "blueberries", "cinnamon"],
        "dietary_tags": ["VEGETARIAN"],
    },
    {
        "name": "Almond Croissant",
        "description": "Butter croissant filled with almond cream.",
        "price_cents": 525,
        "category": "Pastry",
        "sugar_grams": "21.0",
        "carbs_grams": "43.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["croissant dough", "almonds", "butter"],
        "dietary_tags": ["VEGETARIAN"],
    },
    {
        "name": "Chocolate Chip Cookie",
        "description": "Soft-baked cookie with semi-sweet chocolate chips.",
        "price_cents": 350,
        "category": "Pastry",
        "sugar_grams": "22.0",
        "carbs_grams": "38.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["cookie dough", "chocolate chips"],
        "dietary_tags": ["VEGETARIAN"],
    },
    {
        "name": "Peanut Butter Cookie",
        "description": "Soft-baked cookie with peanut butter and brown sugar.",
        "price_cents": 375,
        "category": "Pastry",
        "sugar_grams": "24.0",
        "carbs_grams": "39.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["cookie dough", "peanut butter"],
        "dietary_tags": ["VEGETARIAN"],
    },
    {
        "name": "Avocado Toast",
        "description": "Sourdough toast with avocado, tomato, and chili flakes.",
        "price_cents": 875,
        "category": "Sandwich",
        "sugar_grams": "4.0",
        "carbs_grams": "45.0",
        "is_available": True,
        "allergen_data_complete": True,
        "ingredients": ["sourdough", "avocado", "tomato", "chili flakes"],
        "dietary_tags": ["VEGAN", "VEGETARIAN"],
    },
    {
        "name": "Turkey Pesto Panini",
        "description": "Turkey, mozzarella, tomato, and pesto on grilled sourdough.",
        "price_cents": 1095,
        "category": "Sandwich",
        "sugar_grams": "6.0",
        "carbs_grams": "48.0",
        "is_available": True,
        "allergen_data_complete": False,
        "ingredients": ["sourdough", "turkey", "mozzarella", "tomato", "supplier pesto"],
        "dietary_tags": [],
    },
    {
        "name": "Seasonal Berry Danish",
        "description": "Flaky pastry with rotating berry filling.",
        "price_cents": 495,
        "category": "Pastry",
        "sugar_grams": "26.0",
        "carbs_grams": "46.0",
        "is_available": True,
        "allergen_data_complete": False,
        "ingredients": ["supplier danish dough", "seasonal berry filling"],
        "dietary_tags": ["VEGETARIAN"],
    },
)


def to_decimal(value: str | None) -> Decimal | None:
    """Convert fixture text into Decimal nutrition values.

    Args:
        value (str | None):
            Value value required to perform this operation.

    Returns:
        Decimal | None:
            Decimal nutrition value, or None when the source value is unknown.
    """
    return Decimal(value) if value is not None else None


async def seed_database(session: AsyncSession) -> bool:
    """Insert the deterministic legacy menu fixture into a test database.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.

    Returns:
        bool:
            True when fixture rows were inserted; false when they already existed.
    """
    existing_tenant = await session.scalar(select(Tenant).where(Tenant.name == TENANT_NAME))
    if existing_tenant is not None:
        return False

    allergens = {code: Allergen(code=code, name=name) for code, name in ALLERGENS}
    dietary_tags = {code: DietaryTag(code=code, name=name) for code, name in DIETARY_TAGS}

    ingredient_names = {
        ingredient_name
        for item in MENU_ITEMS
        for ingredient_name in item["ingredients"]
    } | set(INGREDIENT_ALLERGEN_CODES)
    ingredients = {name: Ingredient(name=name) for name in sorted(ingredient_names)}

    for ingredient_name, allergen_codes in INGREDIENT_ALLERGEN_CODES.items():
        ingredients[ingredient_name].allergens = [allergens[code] for code in allergen_codes]

    tenant = Tenant(name=TENANT_NAME)
    tenant.locations.append(Location(name=LOCATION_NAME))

    for item in MENU_ITEMS:
        menu_item = MenuItem(
            tenant=tenant,
            name=item["name"],
            description=item["description"],
            price_cents=item["price_cents"],
            category=item["category"],
            sugar_grams=to_decimal(item["sugar_grams"]),
            carbs_grams=to_decimal(item["carbs_grams"]),
            is_available=item["is_available"],
            allergen_data_complete=item["allergen_data_complete"],
            ingredients=[ingredients[name] for name in item["ingredients"]],
            dietary_tags=[dietary_tags[code] for code in item["dietary_tags"]],
        )
        session.add(menu_item)

    session.add_all([*allergens.values(), *dietary_tags.values(), *ingredients.values(), tenant])
    await session.commit()
    return True

