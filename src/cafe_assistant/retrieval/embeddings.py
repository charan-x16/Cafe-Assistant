"""Implementation module for embeddings.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

from cafe_assistant.db.models import CatalogItemVariant, MenuItem, PolicyChunk
from cafe_assistant.gateway.model_gateway import EmbeddingProvider
from cafe_assistant.security.injection import neutralize_instruction_patterns


def build_menu_item_embedding_text(item: MenuItem) -> str:
    """Build menu item embedding text.

    Args:
        item (MenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.

    Returns:
        str:
            Constructed value used by the caller for retrieval, tracing, or storage.
    """
    dietary_tags = " ".join(sorted(tag.code for tag in item.dietary_tags))
    ingredients = " ".join(
        sorted(neutralize_instruction_patterns(ingredient.name) for ingredient in item.ingredients)
    )
    return " ".join(
        part
        for part in (
            neutralize_instruction_patterns(item.name),
            neutralize_instruction_patterns(item.description),
            neutralize_instruction_patterns(item.category),
            dietary_tags,
            ingredients,
        )
        if part
    )


def embed_menu_item(item: MenuItem, provider: EmbeddingProvider) -> list[float]:
    """Embed menu item.

    Args:
        item (MenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.
        provider (EmbeddingProvider):
            Optional embedding provider override used by tests or scripts.

    Returns:
        list[float]:
            Value produced for the caller according to the function contract.
    """
    return provider.embed([build_menu_item_embedding_text(item)])[0]


def build_catalog_variant_embedding_text(variant: CatalogItemVariant) -> str:
    """Build neutralized embedding text for a production catalog variant.

    Args:
        variant (CatalogItemVariant):
            Catalog variant row converted into a retrievable menu view.

    Returns:
        str:
            Constructed value used by the caller for retrieval, tracing, or storage.
    """
    item = variant.catalog_item
    category_path = item.category.path if item.category is not None else ""
    dietary = " ".join(
        sorted(
            assertion.dietary_tag.code
            for assertion in item.dietary_assertions
            if assertion.assertion_type in {"SUITABLE", "ADAPTABLE"}
        )
    )
    allergens = " ".join(
        sorted(
            assertion.allergen.code
            for assertion in item.allergen_assertions
            if assertion.assertion_type in {"CONTAINS", "MAY_CONTAIN", "CROSS_CONTACT_RISK"}
        )
    )
    ingredients = " ".join(
        sorted(neutralize_instruction_patterns(ingredient.name) for ingredient in item.ingredients)
    )
    tags = " ".join(sorted(neutralize_instruction_patterns(tag) for tag in item.tags))
    return " ".join(
        part
        for part in (
            neutralize_instruction_patterns(item.display_name),
            "" if variant.name == "Default" else neutralize_instruction_patterns(variant.name),
            neutralize_instruction_patterns(item.description),
            neutralize_instruction_patterns(category_path),
            neutralize_instruction_patterns(variant.serving or ""),
            neutralize_instruction_patterns(variant.temperature or ""),
            neutralize_instruction_patterns(variant.caffeine_level or ""),
            neutralize_instruction_patterns(variant.sweetness_level or ""),
            neutralize_instruction_patterns(variant.spice_level or ""),
            tags,
            dietary,
            allergens,
            ingredients,
        )
        if part
    )


def build_policy_chunk_embedding_text(chunk: PolicyChunk) -> str:
    """Build neutralized embedding text for a policy chunk.

    Args:
        chunk (PolicyChunk):
            Chunk value required to perform this operation.

    Returns:
        str:
            Constructed value used by the caller for retrieval, tracing, or storage.
    """
    return " ".join(
        part
        for part in (
            neutralize_instruction_patterns(chunk.heading_path),
            neutralize_instruction_patterns(chunk.content),
        )
        if part
    )


def embed_catalog_variant(
    variant: CatalogItemVariant,
    provider: EmbeddingProvider,
) -> list[float]:
    """Embed catalog variant.

    Args:
        variant (CatalogItemVariant):
            Catalog variant row converted into a retrievable menu view.
        provider (EmbeddingProvider):
            Optional embedding provider override used by tests or scripts.

    Returns:
        list[float]:
            Value produced for the caller according to the function contract.
    """
    return provider.embed([build_catalog_variant_embedding_text(variant)])[0]


def embed_query(query: str, provider: EmbeddingProvider) -> list[float]:
    """Embed a user query after neutralizing instruction-like text.

    Args:
        query (str):
            User search text or policy question to retrieve against.
        provider (EmbeddingProvider):
            Optional embedding provider override used by tests or scripts.

    Returns:
        list[float]:
            Value produced for the caller according to the function contract.
    """
    return provider.embed([neutralize_instruction_patterns(query)])[0]
