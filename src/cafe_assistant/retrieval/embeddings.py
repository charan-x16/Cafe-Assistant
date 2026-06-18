from __future__ import annotations

from cafe_assistant.db.models import MenuItem
from cafe_assistant.gateway.model_gateway import EmbeddingProvider
from cafe_assistant.security.injection import neutralize_instruction_patterns


def build_menu_item_embedding_text(item: MenuItem) -> str:
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
    return provider.embed([build_menu_item_embedding_text(item)])[0]


def embed_query(query: str, provider: EmbeddingProvider) -> list[float]:
    return provider.embed([neutralize_instruction_patterns(query)])[0]
