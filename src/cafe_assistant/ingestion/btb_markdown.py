"""Parser and importer for By The Brew Markdown menu and policy documents.
Converts source documents into versioned catalog items, variants, assertions, modifiers,
ingredients, source-document records, and policy chunks while preserving safety uncertainty.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import (
    Allergen,
    CatalogItem,
    CatalogItemAllergenAssertion,
    CatalogItemDietaryAssertion,
    CatalogItemNutrition,
    CatalogItemVariant,
    DietaryTag,
    Ingredient,
    Menu,
    MenuCategory,
    MenuImportBatch,
    MenuVersion,
    ModifierGroup,
    ModifierOption,
    ModifierOptionAllergenAssertion,
    ModifierOptionDietaryAssertion,
    PolicyChunk,
    PolicyDocument,
    SourceDocument,
    Tenant,
)

BTB_TENANT_NAME = "By The Brew"
BTB_MENU_NAME = "By The Brew Main Menu"
BTB_MENU_SLUG = "by-the-brew-main"
BTB_SOURCE_DIR = Path("docs/source/by-the-brew")

DOCUMENTS: dict[str, str] = {
    "BTB_Menu_Enhanced.md": "menu_enhanced",
    "BTB_Menu_Attributes.md": "menu_attributes",
    "BTB_Company_Policies.md": "company_policy",
}

ALLERGENS: tuple[tuple[str, str], ...] = (
    ("PEANUT", "Peanut"),
    ("TREE_NUT", "Tree nut"),
    ("DAIRY", "Dairy"),
    ("GLUTEN", "Gluten"),
    ("SOY", "Soy"),
    ("EGG", "Egg"),
    ("FISH", "Fish"),
)

DIETARY_TAGS: tuple[tuple[str, str], ...] = (
    ("VEGAN", "Vegan"),
    ("VEGETARIAN", "Vegetarian"),
    ("GLUTEN_FREE", "Gluten free"),
)

ALLERGEN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "PEANUT": ("peanut",),
    "TREE_NUT": (
        "tree nut",
        "almond",
        "hazelnut",
        "cashew",
        "walnut",
        "pine nut",
        "nutella",
        "nuts",
    ),
    "DAIRY": (
        "dairy",
        "milk",
        "cheese",
        "cream",
        "butter",
        "paneer",
        "lactose",
        "yogurt",
        "gelato",
    ),
    "GLUTEN": (
        "gluten",
        "wheat",
        "flour",
        "bread",
        "pasta",
        "tortilla",
        "pizza dough",
        "crust",
        "batter",
        "breading",
        "cookie",
    ),
    "SOY": ("soy", "soya"),
    "EGG": ("egg", "eggs", "mayonnaise", "mayo"),
    "FISH": ("fish", "anchovy", "anchovies"),
}

FIELD_PATTERN = re.compile(r"^- \*\*(?P<name>[^:*]+):\*\*\s*(?P<value>.*)$")
HEADING_PATTERN = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
PRICE_PATTERN = re.compile(r"(?P<amount>\d{2,4})")


@dataclass(frozen=True, slots=True)
class ParsedVariant:
    """Container for parsed variant behavior and data."""
    name: str
    price_cents: int | None = None
    serving: str | None = None
    temperature: str | None = None
    caffeine_level: str | None = None
    sweetness_level: str | None = None
    spice_level: str | None = None


@dataclass(slots=True)
class ParsedMenuItem:
    """Container for parsed menu item behavior and data."""
    canonical_name: str
    display_name: str
    category_path: str
    description: str
    fields: dict[str, str]
    variants: list[ParsedVariant]
    source_heading_path: str
    sort_order: int
    tags: list[str] = field(default_factory=list)
    ingredients: list[str] = field(default_factory=list)
    allergen_assertions: dict[str, str] = field(default_factory=dict)
    allergen_evidence: dict[str, str] = field(default_factory=dict)
    dietary_assertions: dict[str, str] = field(default_factory=dict)
    dietary_evidence: dict[str, str] = field(default_factory=dict)
    allergen_data_complete: bool = False
    dietary_data_complete: bool = False
    nutrition_data_complete: bool = False
    add_on_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedPolicyChunk:
    """Container for parsed policy chunk behavior and data."""
    heading_path: str
    content: str
    content_hash: str
    chunk_index: int


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Container for import result behavior and data."""
    inserted: bool
    tenant_id: int
    menu_version_id: int
    version: str
    item_count: int
    variant_count: int
    policy_chunk_count: int
    incomplete_allergen_item_count: int


async def import_btb_documents(
    session: AsyncSession,
    *,
    source_dir: Path | str = BTB_SOURCE_DIR,
    tenant_name: str = BTB_TENANT_NAME,
    publish: bool = True,
) -> ImportResult:
    """Import BTB source documents into a versioned production catalog.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        source_dir (Path | str):
            Source dir value required to perform this operation.
        tenant_name (str):
            Tenant name value required to perform this operation.
        publish (bool):
            Publish value required to perform this operation.

    Returns:
        ImportResult:
            Import summary including tenant, menu version, and inserted record counts.
    """
    source_path = Path(source_dir)
    document_texts = _read_documents(source_path)
    combined_hash = _hash_text("\n".join(document_texts[name] for name in sorted(document_texts)))
    version = f"btb-{combined_hash[:12]}"

    tenant = await _get_or_create_tenant(session, tenant_name)
    menu = await _get_or_create_menu(session, tenant)
    existing_version = await _load_menu_version(session, menu.id, version)
    if existing_version is not None:
        counts = await _count_existing_version(existing_version)
        return ImportResult(
            inserted=False,
            tenant_id=tenant.id,
            menu_version_id=existing_version.id,
            version=version,
            item_count=counts["items"],
            variant_count=counts["variants"],
            policy_chunk_count=counts["policy_chunks"],
            incomplete_allergen_item_count=counts["incomplete_allergen_items"],
        )

    source_documents = await _upsert_source_documents(session, tenant, source_path, document_texts)
    source_by_type = {document.document_type: document for document in source_documents}
    await _ensure_reference_taxonomies(session)
    allergens = await _load_allergens(session)
    dietary_tags = await _load_dietary_tags(session)

    enhanced_items = _parse_enhanced_menu(document_texts["BTB_Menu_Enhanced.md"])
    attribute_items = _parse_attributes_menu(document_texts["BTB_Menu_Attributes.md"])
    parsed_items = _merge_menu_sources(enhanced_items, attribute_items)
    policy_chunks = _parse_policy_chunks(document_texts["BTB_Company_Policies.md"])

    batch = MenuImportBatch(
        tenant=tenant,
        status="published" if publish else "staged",
        notes="Deterministic import from By The Brew Markdown source documents.",
        completed_at=datetime.now(tz=UTC),
        validation_report=_build_validation_report(parsed_items, policy_chunks),
        source_documents=source_documents,
    )
    menu_version = MenuVersion(
        menu=menu,
        import_batch=batch,
        version=version,
        status="published" if publish else "staged",
        reviewed_at=datetime.now(tz=UTC) if publish else None,
        published_at=datetime.now(tz=UTC) if publish else None,
    )
    session.add(batch)
    session.add(menu_version)
    await session.flush()

    category_by_path = await _create_categories(session, menu_version, parsed_items)
    await _create_catalog_items(
        session,
        menu_version=menu_version,
        parsed_items=parsed_items,
        category_by_path=category_by_path,
        allergens=allergens,
        dietary_tags=dietary_tags,
        source_document=source_by_type["menu_attributes"],
    )
    await _create_policy_document(
        session,
        tenant=tenant,
        source_document=source_by_type["company_policy"],
        policy_chunks=policy_chunks,
    )

    await session.commit()
    return ImportResult(
        inserted=True,
        tenant_id=tenant.id,
        menu_version_id=menu_version.id,
        version=version,
        item_count=len(parsed_items),
        variant_count=sum(len(item.variants) for item in parsed_items),
        policy_chunk_count=len(policy_chunks),
        incomplete_allergen_item_count=sum(
            1 for item in parsed_items if not item.allergen_data_complete
        ),
    )


def _read_documents(source_dir: Path) -> dict[str, str]:
    """Handle read documents.

    Args:
        source_dir (Path):
            Source dir value required to perform this operation.

    Returns:
        dict[str, str]:
            Value produced for the caller according to the function contract.
    """
    documents: dict[str, str] = {}
    for filename in DOCUMENTS:
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing BTB source document: {path}")
        documents[filename] = path.read_text(encoding="utf-8")
    return documents


async def _get_or_create_tenant(session: AsyncSession, tenant_name: str) -> Tenant:
    """Return or create tenant.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_name (str):
            Tenant name value required to perform this operation.

    Returns:
        Tenant:
            Value produced for the caller according to the function contract.
    """
    tenant = await session.scalar(select(Tenant).where(Tenant.name == tenant_name))
    if tenant is not None:
        return tenant
    tenant = Tenant(name=tenant_name)
    session.add(tenant)
    await session.flush()
    return tenant


async def _get_or_create_menu(session: AsyncSession, tenant: Tenant) -> Menu:
    """Return or create menu.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant (Tenant):
            Tenant row that owns imported catalog and policy data.

    Returns:
        Menu:
            Value produced for the caller according to the function contract.
    """
    menu = await session.scalar(
        select(Menu).where(Menu.tenant_id == tenant.id).where(Menu.slug == BTB_MENU_SLUG)
    )
    if menu is not None:
        return menu
    menu = Menu(tenant=tenant, name=BTB_MENU_NAME, slug=BTB_MENU_SLUG)
    session.add(menu)
    await session.flush()
    return menu


async def _load_menu_version(
    session: AsyncSession,
    menu_id: int,
    version: str,
) -> MenuVersion | None:
    """Load menu version.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        menu_id (int):
            Menu id value required to perform this operation.
        version (str):
            Version value required to perform this operation.

    Returns:
        MenuVersion | None:
            Loaded records or projected domain values matching the requested scope.
    """
    return await session.scalar(
        select(MenuVersion)
        .where(MenuVersion.menu_id == menu_id)
        .where(MenuVersion.version == version)
        .options(
            selectinload(MenuVersion.items).selectinload(CatalogItem.variants),
            selectinload(MenuVersion.items),
        )
    )


async def _count_existing_version(menu_version: MenuVersion) -> dict[str, int]:
    """Count existing version.

    Args:
        menu_version (MenuVersion):
            Versioned menu catalog row being populated or queried.

    Returns:
        dict[str, int]:
            Value produced for the caller according to the function contract.
    """
    items = list(menu_version.items)
    return {
        "items": len(items),
        "variants": sum(len(item.variants) for item in items),
        "policy_chunks": 0,
        "incomplete_allergen_items": sum(1 for item in items if not item.allergen_data_complete),
    }


async def _upsert_source_documents(
    session: AsyncSession,
    tenant: Tenant,
    source_dir: Path,
    document_texts: dict[str, str],
) -> list[SourceDocument]:
    """Insert or update source documents.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant (Tenant):
            Tenant row that owns imported catalog and policy data.
        source_dir (Path):
            Source dir value required to perform this operation.
        document_texts (dict[str, str]):
            Markdown source text keyed by source filename.

    Returns:
        list[SourceDocument]:
            Value produced for the caller according to the function contract.
    """
    documents: list[SourceDocument] = []
    for filename, document_type in DOCUMENTS.items():
        content_hash = _hash_text(document_texts[filename])
        document = await session.scalar(
            select(SourceDocument)
            .where(SourceDocument.tenant_id == tenant.id)
            .where(SourceDocument.content_hash == content_hash)
        )
        if document is None:
            document = SourceDocument(
                tenant=tenant,
                document_type=document_type,
                filename=filename,
                path=str((source_dir / filename).as_posix()),
                version=_extract_document_version(document_texts[filename]),
                content_hash=content_hash,
            )
            session.add(document)
            await session.flush()
        documents.append(document)
    return documents


async def _ensure_reference_taxonomies(session: AsyncSession) -> None:
    """Ensure reference taxonomies.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    existing_allergens = {
        code
        for code in await session.scalars(
            select(Allergen.code).where(Allergen.code.in_([a[0] for a in ALLERGENS]))
        )
    }
    existing_dietary_tags = {
        code
        for code in await session.scalars(
            select(DietaryTag.code).where(DietaryTag.code.in_([tag[0] for tag in DIETARY_TAGS]))
        )
    }
    for code, name in ALLERGENS:
        if code not in existing_allergens:
            session.add(Allergen(code=code, name=name))
    for code, name in DIETARY_TAGS:
        if code not in existing_dietary_tags:
            session.add(DietaryTag(code=code, name=name))
    await session.flush()


async def _load_allergens(session: AsyncSession) -> dict[str, Allergen]:
    """Load allergens.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.

    Returns:
        dict[str, Allergen]:
            Loaded records or projected domain values matching the requested scope.
    """
    return {
        allergen.code: allergen
        for allergen in await session.scalars(
            select(Allergen).where(Allergen.code.in_([a[0] for a in ALLERGENS]))
        )
    }


async def _load_dietary_tags(session: AsyncSession) -> dict[str, DietaryTag]:
    """Load dietary tags.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.

    Returns:
        dict[str, DietaryTag]:
            Loaded records or projected domain values matching the requested scope.
    """
    return {
        tag.code: tag
        for tag in await session.scalars(
            select(DietaryTag).where(DietaryTag.code.in_([tag[0] for tag in DIETARY_TAGS]))
        )
    }


def _parse_enhanced_menu(text_value: str) -> dict[str, ParsedMenuItem]:
    """Parse enhanced menu.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        dict[str, ParsedMenuItem]:
            Parsed values extracted from the source text or structured payload.
    """
    headings: dict[int, str] = {}
    current_title: str | None = None
    current_level = 0
    current_fields: dict[str, str] = {}
    order = 0
    parsed: dict[str, ParsedMenuItem] = {}

    def flush() -> None:
        """Handle flush.

        Args:
            None.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        nonlocal order, current_title, current_fields, current_level
        if current_title is None or "Price" not in current_fields:
            current_title = None
            current_fields = {}
            return
        category_path = _category_path(headings, current_level)
        canonical_name = _canonical_item_name(current_title)
        variants = _variants_from_price(current_fields.get("Price", ""), current_fields)
        order += 1
        parsed[canonical_name] = ParsedMenuItem(
            canonical_name=canonical_name,
            display_name=current_title,
            category_path=category_path,
            description=current_fields.get("Description", ""),
            fields=dict(current_fields),
            variants=variants,
            source_heading_path=" > ".join(
                headings[level] for level in sorted(headings) if level < current_level
            ),
            sort_order=order,
            tags=_split_tags(current_fields.get("Tags", "")),
            dietary_data_complete="Dietary Tags" in current_fields,
            add_on_fields={
                key: value
                for key, value in current_fields.items()
                if "Add-Ons" in key or "Toppings" in key
            },
        )
        current_title = None
        current_fields = {}

    for raw_line in text_value.splitlines():
        heading_match = HEADING_PATTERN.match(raw_line)
        if heading_match:
            flush()
            level = len(heading_match.group("level"))
            title = _clean_text(heading_match.group("title"))
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = title
            current_title = title if level in {3, 4} else None
            current_level = level
            current_fields = {}
            continue

        field_match = FIELD_PATTERN.match(raw_line)
        if field_match and current_title is not None:
            current_fields[_clean_text(field_match.group("name"))] = _clean_text(
                field_match.group("value")
            )

    flush()
    return parsed


def _parse_attributes_menu(text_value: str) -> dict[str, list[dict[str, str]]]:
    """Parse attributes menu.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        dict[str, list[dict[str, str]]]:
            Parsed values extracted from the source text or structured payload.
    """
    current_title: str | None = None
    current_fields: dict[str, str] = {}
    parsed: dict[str, list[dict[str, str]]] = {}

    def flush() -> None:
        """Handle flush.

        Args:
            None.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        nonlocal current_title, current_fields
        if current_title is None or not current_fields:
            current_title = None
            current_fields = {}
            return
        canonical_name = _canonical_item_name(current_title)
        fields = dict(current_fields)
        fields["__title"] = current_title
        parsed.setdefault(canonical_name, []).append(fields)
        current_title = None
        current_fields = {}

    for raw_line in text_value.splitlines():
        heading_match = HEADING_PATTERN.match(raw_line)
        if heading_match:
            flush()
            level = len(heading_match.group("level"))
            current_title = _clean_text(heading_match.group("title")) if level == 3 else None
            current_fields = {}
            continue

        field_match = FIELD_PATTERN.match(raw_line)
        if field_match and current_title is not None:
            current_fields[_clean_text(field_match.group("name"))] = _clean_text(
                field_match.group("value")
            )
    flush()
    return parsed


def _merge_menu_sources(
    enhanced_items: dict[str, ParsedMenuItem],
    attribute_items: dict[str, list[dict[str, str]]],
) -> list[ParsedMenuItem]:
    """Merge menu sources.

    Args:
        enhanced_items (dict[str, ParsedMenuItem]):
            Menu items parsed from the enhanced menu document.
        attribute_items (dict[str, list[dict[str, str]]]):
            Menu attributes parsed from the attributes document.

    Returns:
        list[ParsedMenuItem]:
            Value produced for the caller according to the function contract.
    """
    merged_items: list[ParsedMenuItem] = []
    for canonical_name, item in enhanced_items.items():
        attributes = attribute_items.get(canonical_name, [])
        if attributes:
            _merge_attributes_into_item(item, attributes)
        else:
            item.allergen_data_complete = False
        merged_items.append(item)
    merged_items.sort(key=lambda item: item.sort_order)
    return merged_items


def _merge_attributes_into_item(
    item: ParsedMenuItem,
    attributes: list[dict[str, str]],
) -> None:
    """Merge attributes into item.

    Args:
        item (ParsedMenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.
        attributes (list[dict[str, str]]):
            Attributes value required to perform this operation.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    first = attributes[0]
    item.ingredients = _split_ingredients(first.get("Ingredients", ""))
    if not item.tags:
        item.tags = _split_tags(first.get("Taste", ""))
    item.allergen_assertions, item.allergen_evidence = _allergen_assertions(attributes)
    item.dietary_assertions, item.dietary_evidence = _dietary_assertions(item, attributes)
    item.allergen_data_complete = _allergen_data_complete(attributes)
    item.dietary_data_complete = bool(item.dietary_assertions)

    attribute_variants = [_variant_name_from_title(fields["__title"]) for fields in attributes]
    if any(name != "Default" for name in attribute_variants):
        prices_by_name = {variant.name: variant.price_cents for variant in item.variants}
        item.variants = [
            ParsedVariant(
                name=variant_name,
                price_cents=prices_by_name.get(variant_name) or _single_price(item.variants),
                serving=fields.get("Serving"),
                temperature=fields.get("Temperature"),
                caffeine_level=fields.get("Caffeine Level"),
                sweetness_level=fields.get("Sweetness"),
                spice_level=fields.get("Spice Level"),
            )
            for variant_name, fields in zip(attribute_variants, attributes, strict=True)
        ]
    else:
        item.variants = [
            ParsedVariant(
                name=variant.name,
                price_cents=variant.price_cents,
                serving=first.get("Serving") or variant.serving,
                temperature=first.get("Temperature") or variant.temperature,
                caffeine_level=first.get("Caffeine Level") or variant.caffeine_level,
                sweetness_level=first.get("Sweetness") or variant.sweetness_level,
                spice_level=first.get("Spice Level") or variant.spice_level,
            )
            for variant in item.variants
        ]


def _parse_policy_chunks(text_value: str) -> list[ParsedPolicyChunk]:
    """Parse policy chunks.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        list[ParsedPolicyChunk]:
            Parsed values extracted from the source text or structured payload.
    """
    heading_stack: dict[int, str] = {}
    current_heading_path = "Document"
    current_lines: list[str] = []
    chunks: list[ParsedPolicyChunk] = []

    def flush() -> None:
        """Handle flush.

        Args:
            None.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        nonlocal current_lines
        content = "\n".join(line.rstrip() for line in current_lines).strip()
        if not content:
            current_lines = []
            return
        chunks.append(
            ParsedPolicyChunk(
                heading_path=current_heading_path,
                content=content,
                content_hash=_hash_text(f"{current_heading_path}\n{content}"),
                chunk_index=len(chunks),
            )
        )
        current_lines = []

    for raw_line in text_value.splitlines():
        heading_match = HEADING_PATTERN.match(raw_line)
        if heading_match:
            flush()
            level = len(heading_match.group("level"))
            title = _clean_text(heading_match.group("title"))
            heading_stack = {key: value for key, value in heading_stack.items() if key < level}
            heading_stack[level] = title
            current_heading_path = " > ".join(heading_stack[key] for key in sorted(heading_stack))
            continue
        current_lines.append(_clean_text(raw_line))
    flush()
    return chunks


async def _create_categories(
    session: AsyncSession,
    menu_version: MenuVersion,
    parsed_items: list[ParsedMenuItem],
) -> dict[str, MenuCategory]:
    """Create categories.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        menu_version (MenuVersion):
            Versioned menu catalog row being populated or queried.
        parsed_items (list[ParsedMenuItem]):
            Parsed catalog items derived from BTB source documents.

    Returns:
        dict[str, MenuCategory]:
            Value produced for the caller according to the function contract.
    """
    category_by_path: dict[str, MenuCategory] = {}
    order = 0
    for item in parsed_items:
        parts = [part.strip() for part in item.category_path.split(">") if part.strip()]
        parent: MenuCategory | None = None
        path_parts: list[str] = []
        for part in parts:
            path_parts.append(part)
            path = " > ".join(path_parts)
            if path in category_by_path:
                parent = category_by_path[path]
                continue
            order += 1
            category = MenuCategory(
                menu_version=menu_version,
                parent=parent,
                name=part,
                path=path,
                sort_order=order,
            )
            session.add(category)
            await session.flush()
            category_by_path[path] = category
            parent = category
    return category_by_path


async def _create_catalog_items(
    session: AsyncSession,
    *,
    menu_version: MenuVersion,
    parsed_items: list[ParsedMenuItem],
    category_by_path: dict[str, MenuCategory],
    allergens: dict[str, Allergen],
    dietary_tags: dict[str, DietaryTag],
    source_document: SourceDocument,
) -> None:
    """Create catalog items.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        menu_version (MenuVersion):
            Versioned menu catalog row being populated or queried.
        parsed_items (list[ParsedMenuItem]):
            Parsed catalog items derived from BTB source documents.
        category_by_path (dict[str, MenuCategory]):
            Catalog categories keyed by hierarchical category path.
        allergens (dict[str, Allergen]):
            Allergen reference rows keyed by allergen code.
        dietary_tags (dict[str, DietaryTag]):
            Dietary tag reference rows keyed by dietary code.
        source_document (SourceDocument):
            Source document value required to perform this operation.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    plant_milk_group = _plant_milk_group(menu_version, allergens, dietary_tags)
    session.add(plant_milk_group)
    ingredient_cache = await _load_ingredient_cache(session, parsed_items)

    for item in parsed_items:
        catalog_item = CatalogItem(
            menu_version_id=menu_version.id,
            category_id=(
                category_by_path[item.category_path].id
                if item.category_path in category_by_path
                else None
            ),
            source_document_id=source_document.id,
            canonical_name=item.canonical_name,
            display_name=item.display_name,
            description=item.description,
            source_heading_path=item.source_heading_path,
            tags=item.tags,
            allergen_data_complete=item.allergen_data_complete,
            dietary_data_complete=item.dietary_data_complete,
            sort_order=item.sort_order,
        )
        with session.no_autoflush:
            catalog_item.ingredients = [
                await _get_or_create_ingredient(session, ingredient_name, ingredient_cache)
                for ingredient_name in item.ingredients
            ]
        catalog_item.variants = [
            CatalogItemVariant(
                name=variant.name,
                serving=variant.serving,
                temperature=variant.temperature,
                price_cents=variant.price_cents,
                currency="INR",
                caffeine_level=variant.caffeine_level,
                sweetness_level=variant.sweetness_level,
                spice_level=variant.spice_level,
                is_default=index == 0,
                sort_order=index,
                nutrition=CatalogItemNutrition(
                    data_complete=item.nutrition_data_complete,
                    notes="Nutrition values were not exact in the source Markdown.",
                ),
            )
            for index, variant in enumerate(item.variants)
        ]
        catalog_item.allergen_assertions = [
            CatalogItemAllergenAssertion(
                allergen=allergens[code],
                assertion_type=assertion_type,
                evidence=item.allergen_evidence.get(code),
                source_document=source_document,
            )
            for code, assertion_type in sorted(item.allergen_assertions.items())
            if code in allergens
        ]
        catalog_item.dietary_assertions = [
            CatalogItemDietaryAssertion(
                dietary_tag=dietary_tags[code],
                assertion_type=assertion_type,
                evidence=item.dietary_evidence.get(code),
                source_document=source_document,
            )
            for code, assertion_type in sorted(item.dietary_assertions.items())
            if code in dietary_tags
        ]
        if _has_plant_milk_upgrade(item):
            catalog_item.modifier_groups.append(plant_milk_group)
        session.add(catalog_item)
        await session.flush()
        await _create_item_add_on_groups(session, menu_version, catalog_item, item)


def _plant_milk_group(
    menu_version: MenuVersion,
    allergens: dict[str, Allergen],
    dietary_tags: dict[str, DietaryTag],
) -> ModifierGroup:
    """Handle plant milk group.

    Args:
        menu_version (MenuVersion):
            Versioned menu catalog row being populated or queried.
        allergens (dict[str, Allergen]):
            Allergen reference rows keyed by allergen code.
        dietary_tags (dict[str, DietaryTag]):
            Dietary tag reference rows keyed by dietary code.

    Returns:
        ModifierGroup:
            Value produced for the caller according to the function contract.
    """
    group = ModifierGroup(
        menu_version=menu_version,
        name="Plant-Based Milk Upgrade",
        selection_type="optional_single",
        min_select=0,
        max_select=1,
        sort_order=0,
    )
    group.options = [
        _plant_milk_option(
            name="Soya Milk",
            allergen_code="SOY",
            assertion_type="CONTAINS",
            allergens=allergens,
            dietary_tags=dietary_tags,
            allergen_data_complete=True,
            sort_order=1,
        ),
        _plant_milk_option(
            name="Oat Milk",
            allergen_code="GLUTEN",
            assertion_type="MAY_CONTAIN",
            allergens=allergens,
            dietary_tags=dietary_tags,
            allergen_data_complete=False,
            sort_order=2,
            note="Confirm gluten-free oat milk.",
        ),
        _plant_milk_option(
            name="Almond Milk",
            allergen_code="TREE_NUT",
            assertion_type="CONTAINS",
            allergens=allergens,
            dietary_tags=dietary_tags,
            allergen_data_complete=True,
            sort_order=3,
        ),
    ]
    return group


def _plant_milk_option(
    *,
    name: str,
    allergen_code: str,
    assertion_type: str,
    allergens: dict[str, Allergen],
    dietary_tags: dict[str, DietaryTag],
    allergen_data_complete: bool,
    sort_order: int,
    note: str | None = None,
) -> ModifierOption:
    """Handle plant milk option.

    Args:
        name (str):
            Name value required to perform this operation.
        allergen_code (str):
            Allergen code value required to perform this operation.
        assertion_type (str):
            Assertion type value required to perform this operation.
        allergens (dict[str, Allergen]):
            Allergen reference rows keyed by allergen code.
        dietary_tags (dict[str, DietaryTag]):
            Dietary tag reference rows keyed by dietary code.
        allergen_data_complete (bool):
            Allergen data complete value required to perform this operation.
        sort_order (int):
            Sort order value required to perform this operation.
        note (str | None):
            Note value required to perform this operation.

    Returns:
        ModifierOption:
            Value produced for the caller according to the function contract.
    """
    option = ModifierOption(
        name=name,
        price_delta_cents=6000,
        allergen_data_complete=allergen_data_complete,
        dietary_data_complete=True,
        sort_order=sort_order,
        metadata_json={
            "source": "company_policy",
            **({"note": note} if note else {}),
        },
    )
    option.allergen_assertions = [
        ModifierOptionAllergenAssertion(
            allergen=allergens[allergen_code],
            assertion_type=assertion_type,
            evidence=f"{name} selected as plant-based milk upgrade.",
        )
    ]
    option.dietary_assertions = [
        ModifierOptionDietaryAssertion(
            dietary_tag=dietary_tags["VEGAN"],
            assertion_type="SUITABLE",
            evidence="Plant-based milk upgrade.",
        ),
        ModifierOptionDietaryAssertion(
            dietary_tag=dietary_tags["VEGETARIAN"],
            assertion_type="SUITABLE",
            evidence="Plant-based milk upgrade.",
        ),
    ]
    return option


async def _create_item_add_on_groups(
    session: AsyncSession,
    menu_version: MenuVersion,
    catalog_item: CatalogItem,
    item: ParsedMenuItem,
) -> None:
    """Create item add on groups.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        menu_version (MenuVersion):
            Versioned menu catalog row being populated or queried.
        catalog_item (CatalogItem):
            Catalog item value required to perform this operation.
        item (ParsedMenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    for field_name, raw_value in item.add_on_fields.items():
        options = _parse_modifier_options(raw_value)
        if not options:
            continue
        group = ModifierGroup(
            menu_version=menu_version,
            name=f"{catalog_item.display_name} {field_name}",
            selection_type="optional_multi",
            min_select=0,
            max_select=None,
            sort_order=item.sort_order,
        )
        group.options = [
            ModifierOption(
                name=name,
                price_delta_cents=price_cents,
                allergen_data_complete=False,
                dietary_data_complete=False,
                sort_order=index,
            )
            for index, (name, price_cents) in enumerate(options)
        ]
        group.catalog_items.append(catalog_item)
        session.add(group)


async def _create_policy_document(
    session: AsyncSession,
    *,
    tenant: Tenant,
    source_document: SourceDocument,
    policy_chunks: list[ParsedPolicyChunk],
) -> None:
    """Create policy document.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant (Tenant):
            Tenant row that owns imported catalog and policy data.
        source_document (SourceDocument):
            Source document value required to perform this operation.
        policy_chunks (list[ParsedPolicyChunk]):
            Policy chunks parsed from the company policy document.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    policy_document = PolicyDocument(
        tenant=tenant,
        source_document=source_document,
        title="By The Brew Company Policies",
        version=source_document.version,
        chunks=[
            PolicyChunk(
                heading_path=chunk.heading_path,
                content=chunk.content,
                content_hash=chunk.content_hash,
                chunk_index=chunk.chunk_index,
            )
            for chunk in policy_chunks
        ],
    )
    session.add(policy_document)


async def _load_ingredient_cache(
    session: AsyncSession,
    parsed_items: list[ParsedMenuItem],
) -> dict[str, Ingredient]:
    """Load ingredient cache.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        parsed_items (list[ParsedMenuItem]):
            Parsed catalog items derived from BTB source documents.

    Returns:
        dict[str, Ingredient]:
            Loaded records or projected domain values matching the requested scope.
    """
    ingredient_names = {
        ingredient_name.strip().lower()
        for item in parsed_items
        for ingredient_name in item.ingredients
        if ingredient_name.strip()
    }
    if not ingredient_names:
        return {}
    existing = await session.scalars(
        select(Ingredient).where(Ingredient.name.in_(ingredient_names))
    )
    return {ingredient.name: ingredient for ingredient in existing}


async def _get_or_create_ingredient(
    session: AsyncSession,
    name: str,
    ingredient_cache: dict[str, Ingredient],
) -> Ingredient:
    """Return or create ingredient.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        name (str):
            Name value required to perform this operation.
        ingredient_cache (dict[str, Ingredient]):
            Ingredient rows keyed by normalized ingredient name.

    Returns:
        Ingredient:
            Value produced for the caller according to the function contract.
    """
    normalized = name.strip().lower()
    if normalized in ingredient_cache:
        return ingredient_cache[normalized]
    ingredient = Ingredient(name=normalized)
    session.add(ingredient)
    await session.flush()
    ingredient_cache[normalized] = ingredient
    return ingredient


def _category_path(headings: dict[int, str], item_level: int) -> str:
    """Handle category path.

    Args:
        headings (dict[int, str]):
            Headings value required to perform this operation.
        item_level (int):
            Item level value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    if item_level == 4 and 2 in headings and 3 in headings:
        return f"{headings[2]} > {headings[3]}"
    return headings.get(2, "Uncategorized")


def _variants_from_price(price_text: str, fields: dict[str, str]) -> list[ParsedVariant]:
    """Handle variants from price.

    Args:
        price_text (str):
            Raw price text parsed into catalog variants.
        fields (dict[str, str]):
            Parsed Markdown fields for a menu item or modifier option.

    Returns:
        list[ParsedVariant]:
            Value produced for the caller according to the function contract.
    """
    amounts = [int(match.group("amount")) * 100 for match in PRICE_PATTERN.finditer(price_text)]
    serving = fields.get("Serving")
    if len(amounts) >= 2 and "hot" in price_text.lower() and "iced" in price_text.lower():
        return [
            ParsedVariant(name="Hot", price_cents=amounts[0], serving=serving, temperature="Hot"),
            ParsedVariant(name="Iced", price_cents=amounts[1], serving=serving, temperature="Cold"),
        ]
    if len(amounts) == 1:
        variant_name = _variant_name_from_serving(serving or "")
        return [ParsedVariant(name=variant_name, price_cents=amounts[0], serving=serving)]
    return [ParsedVariant(name="Default", price_cents=None, serving=serving)]


def _variant_name_from_serving(serving: str) -> str:
    """Handle variant name from serving.

    Args:
        serving (str):
            Serving value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    lowered = serving.lower()
    if "hot only" in lowered:
        return "Hot"
    if "iced only" in lowered or "cold" in lowered:
        return "Iced"
    return "Default"


def _variant_name_from_title(title: str) -> str:
    """Handle variant name from title.

    Args:
        title (str):
            Title value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    lowered = title.lower()
    if "(hot)" in lowered:
        return "Hot"
    if "(iced)" in lowered:
        return "Iced"
    return "Default"


def _single_price(variants: list[ParsedVariant]) -> int | None:
    """Handle single price.

    Args:
        variants (list[ParsedVariant]):
            Variants value required to perform this operation.

    Returns:
        int | None:
            Value produced for the caller according to the function contract.
    """
    for variant in variants:
        if variant.price_cents is not None:
            return variant.price_cents
    return None


def _canonical_item_name(title: str) -> str:
    """Handle canonical item name.

    Args:
        title (str):
            Title value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    cleaned = re.sub(r"\s*\((hot|iced|non-alcoholic)\)\s*$", "", title, flags=re.IGNORECASE)
    return _clean_text(cleaned)


def _split_tags(raw_value: str) -> list[str]:
    """Handle split tags.

    Args:
        raw_value (str):
            Raw value value required to perform this operation.

    Returns:
        list[str]:
            Value produced for the caller according to the function contract.
    """
    return [
        _clean_text(value).lower()
        for value in re.split(r"[,/]", raw_value)
        if _clean_text(value)
    ]


def _split_ingredients(raw_value: str) -> list[str]:
    """Handle split ingredients.

    Args:
        raw_value (str):
            Raw value value required to perform this operation.

    Returns:
        list[str]:
            Value produced for the caller according to the function contract.
    """
    if not raw_value:
        return []
    simplified = raw_value.replace(" and ", ", ")
    return [
        _clean_text(value).lower()
        for value in simplified.split(",")
        if _clean_text(value)
    ]


def _allergen_assertions(
    attributes: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Handle allergen assertions.

    Args:
        attributes (list[dict[str, str]]):
            Attributes value required to perform this operation.

    Returns:
        tuple[dict[str, str], dict[str, str]]:
            Value produced for the caller according to the function contract.
    """
    assertions: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for fields in attributes:
        for code in _allergen_codes_from_text(fields.get("Does Not Contain", "")):
            assertions.setdefault(code, "DOES_NOT_CONTAIN")
            evidence.setdefault(code, fields.get("Does Not Contain", ""))
        for code in _allergen_codes_from_text(fields.get("Possible Allergens", "")):
            possible_text = fields.get("Possible Allergens", "")
            if "none typical" in possible_text.lower():
                continue
            assertions[code] = _stronger_assertion(assertions.get(code), "MAY_CONTAIN")
            evidence[code] = possible_text
        for code in _allergen_codes_from_text(fields.get("Contains", "")):
            assertions[code] = _stronger_assertion(assertions.get(code), "CONTAINS")
            evidence[code] = fields.get("Contains", "")
    return assertions, evidence


def _dietary_assertions(
    item: ParsedMenuItem,
    attributes: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Handle dietary assertions.

    Args:
        item (ParsedMenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.
        attributes (list[dict[str, str]]):
            Attributes value required to perform this operation.

    Returns:
        tuple[dict[str, str], dict[str, str]]:
            Value produced for the caller according to the function contract.
    """
    assertions: dict[str, str] = {}
    evidence: dict[str, str] = {}
    dietary_text = " ".join(
        [item.fields.get("Dietary Tags", ""), *(fields.get("Dietary", "") for fields in attributes)]
    )
    normalized = dietary_text.lower()
    if "vegetarian" in normalized or "vegan" in normalized:
        assertions["VEGETARIAN"] = "SUITABLE"
        evidence["VEGETARIAN"] = dietary_text
    if "vegan" in normalized and "not vegan" not in normalized and "vegan with" not in normalized:
        assertions["VEGAN"] = "SUITABLE"
        evidence["VEGAN"] = dietary_text
    elif _has_plant_milk_upgrade(item):
        assertions["VEGAN"] = "ADAPTABLE"
        evidence["VEGAN"] = item.fields.get("Vegan Note") or item.fields.get("Vegan Option", "")
    if "gluten-free" in normalized or "gluten free" in normalized:
        assertions["GLUTEN_FREE"] = "SUITABLE"
        evidence["GLUTEN_FREE"] = dietary_text
    return assertions, evidence


def _allergen_codes_from_text(text_value: str) -> set[str]:
    """Handle allergen codes from text.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        set[str]:
            Value produced for the caller according to the function contract.
    """
    normalized = text_value.lower()
    if "none typical" in normalized:
        return set()
    codes: set[str] = set()
    for code, keywords in ALLERGEN_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            codes.add(code)
    return codes


def _stronger_assertion(existing: str | None, new: str) -> str:
    """Handle stronger assertion.

    Args:
        existing (str | None):
            Existing value required to perform this operation.
        new (str):
            New value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    priority = {
        None: 0,
        "DOES_NOT_CONTAIN": 1,
        "UNKNOWN": 2,
        "CROSS_CONTACT_RISK": 3,
        "MAY_CONTAIN": 4,
        "CONTAINS": 5,
    }
    return new if priority[new] >= priority[existing] else existing or new


def _allergen_data_complete(attributes: list[dict[str, str]]) -> bool:
    """Handle allergen data complete.

    Args:
        attributes (list[dict[str, str]]):
            Attributes value required to perform this operation.

    Returns:
        bool:
            Value produced for the caller according to the function contract.
    """
    text_value = " ".join(
        fields.get(key, "")
        for fields in attributes
        for key in ("Contains", "Possible Allergens", "Does Not Contain", "Not Enough Data")
    ).lower()
    risky_unknown_terms = (
        "confirm",
        "composition",
        "brand-dependent",
        "specific nut",
        "cashew",
        "egg content",
        "soy",
        "nut allergy",
        "contains dairy",
        "whether",
        "depends",
    )
    if any(term in text_value for term in risky_unknown_terms):
        return False
    return bool(text_value.strip())


def _has_plant_milk_upgrade(item: ParsedMenuItem) -> bool:
    """Handle has plant milk upgrade.

    Args:
        item (ParsedMenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.

    Returns:
        bool:
            Value produced for the caller according to the function contract.
    """
    fields = " ".join(item.fields.values()).lower()
    return "oat" in fields and "almond" in fields and "vegan" in fields


def _parse_modifier_options(raw_value: str) -> list[tuple[str, int]]:
    """Parse modifier options.

    Args:
        raw_value (str):
            Raw value value required to perform this operation.

    Returns:
        list[tuple[str, int]]:
            Parsed values extracted from the source text or structured payload.
    """
    normalized = raw_value.replace("Â·", "/").replace("·", "/")
    options: list[tuple[str, int]] = []
    for segment in normalized.split("/"):
        cleaned = _clean_text(segment)
        if not cleaned:
            continue
        amounts = [int(match.group("amount")) for match in PRICE_PATTERN.finditer(cleaned)]
        price_cents = amounts[-1] * 100 if amounts else 0
        name = re.sub(r"\([^)]*\)", "", cleaned).strip(" -")
        if name:
            options.append((name, price_cents))
    return options


def _build_validation_report(
    parsed_items: list[ParsedMenuItem],
    policy_chunks: list[ParsedPolicyChunk],
) -> dict[str, object]:
    """Build validation report.

    Args:
        parsed_items (list[ParsedMenuItem]):
            Parsed catalog items derived from BTB source documents.
        policy_chunks (list[ParsedPolicyChunk]):
            Policy chunks parsed from the company policy document.

    Returns:
        dict[str, object]:
            Constructed value used by the caller for retrieval, tracing, or storage.
    """
    incomplete_allergen_items = [
        item.display_name for item in parsed_items if not item.allergen_data_complete
    ]
    return {
        "item_count": len(parsed_items),
        "variant_count": sum(len(item.variants) for item in parsed_items),
        "policy_chunk_count": len(policy_chunks),
        "incomplete_allergen_item_count": len(incomplete_allergen_items),
        "incomplete_allergen_items": incomplete_allergen_items[:100],
        "importer": "btb_markdown_v1",
        "safety_note": "Incomplete or confirm-with-staff allergen data remains unsafe.",
    }


def _extract_document_version(text_value: str) -> str:
    """Handle extract document version.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    for line in text_value.splitlines()[:20]:
        if "Document Version" in line:
            return _clean_text(line.split(":", 1)[-1]).strip("* ")
    return "unversioned"


def _hash_text(text_value: str) -> str:
    """Hash text.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def _clean_text(text_value: str) -> str:
    """Handle clean text.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    replacements = {
        "\u00a0": " ",
        "\u00b7": " - ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    cleaned = text_value
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return re.sub(r"\s+", " ", cleaned).strip()


def import_result_to_dict(result: ImportResult) -> dict[str, Any]:
    """Import result to dict.

    Args:
        result (ImportResult):
            Result value required to perform this operation.

    Returns:
        dict[str, Any]:
            Value produced for the caller according to the function contract.
    """
    return {
        "inserted": result.inserted,
        "tenant_id": result.tenant_id,
        "menu_version_id": result.menu_version_id,
        "version": result.version,
        "item_count": result.item_count,
        "variant_count": result.variant_count,
        "policy_chunk_count": result.policy_chunk_count,
        "incomplete_allergen_item_count": result.incomplete_allergen_item_count,
    }
