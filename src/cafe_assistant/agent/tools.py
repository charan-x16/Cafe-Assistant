from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.repositories.menu_repo import load_menu_item_views_for_tenant
from cafe_assistant.domain.dietary import (
    AllergenCode,
    CustomerRestrictions,
    DietaryMode,
    MenuItemView,
    filter_safe_items,
)
from cafe_assistant.gateway.model_gateway import EmbeddingProvider
from cafe_assistant.observability.tracing import span
from cafe_assistant.retrieval.hybrid import search_menu as retrieval_search_menu


class RestrictionsSchema(BaseModel):
    avoid_allergens: set[AllergenCode] = Field(default_factory=set)
    modes: set[DietaryMode] = Field(default_factory=set)
    prefer_low_sugar: bool = False

    def to_domain(self) -> CustomerRestrictions:
        return CustomerRestrictions(
            avoid_allergens=set(self.avoid_allergens),
            modes=set(self.modes),
            prefer_low_sugar=self.prefer_low_sugar,
        )

    @classmethod
    def from_domain(cls, restrictions: CustomerRestrictions) -> RestrictionsSchema:
        return cls(
            avoid_allergens=set(restrictions.avoid_allergens),
            modes=set(restrictions.modes),
            prefer_low_sugar=restrictions.prefer_low_sugar,
        )


class MenuItemViewSchema(BaseModel):
    id: int
    name: str
    allergen_codes: set[AllergenCode]
    dietary_tags: set[DietaryMode]
    allergen_data_complete: bool
    sugar_grams: float | None

    @classmethod
    def from_domain(cls, item: MenuItemView) -> MenuItemViewSchema:
        return cls(
            id=item.id,
            name=item.name,
            allergen_codes=set(item.allergen_codes),
            dietary_tags=set(item.dietary_tags),
            allergen_data_complete=item.allergen_data_complete,
            sugar_grams=item.sugar_grams,
        )

    def to_domain(self) -> MenuItemView:
        return MenuItemView(
            id=self.id,
            name=self.name,
            allergen_codes=set(self.allergen_codes),
            dietary_tags=set(self.dietary_tags),
            allergen_data_complete=self.allergen_data_complete,
            sugar_grams=self.sugar_grams,
        )


class MenuLookupInput(BaseModel):
    tenant_id: int
    query: str
    limit: int = 5


class SearchMenuInput(BaseModel):
    tenant_id: int
    query: str
    restrictions: RestrictionsSchema
    k: int = 8


class DietaryFilterInput(BaseModel):
    items: list[MenuItemViewSchema]
    restrictions: RestrictionsSchema


class MenuItemsOutput(BaseModel):
    items: list[MenuItemViewSchema]


ToolCallable = Callable[[BaseModel], Awaitable[BaseModel]]


class ToolRegistry:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider
        self._tools: dict[str, ToolCallable] = {
            "menu_lookup": self.menu_lookup,
            "search_menu": self.search_menu,
            "dietary_filter": self.dietary_filter,
        }

    async def call(self, name: str, input_model: BaseModel) -> BaseModel:
        with span("tool.call", tool_name=name):
            return await self._tools[name](input_model)

    async def menu_lookup(self, input_model: BaseModel) -> MenuItemsOutput:
        tool_input = _coerce(input_model, MenuLookupInput)
        views = await load_menu_item_views_for_tenant(self.session, tool_input.tenant_id)
        query = tool_input.query.lower().strip()
        matches = [
            view
            for view in views
            if query in view.name.lower() or view.name.lower() in query
        ][: tool_input.limit]
        with span(
            "tool.menu_lookup",
            tool_name="menu_lookup",
            tenant_id=tool_input.tenant_id,
            retrieved_item_ids=[item.id for item in matches],
        ):
            pass
        return MenuItemsOutput(items=[MenuItemViewSchema.from_domain(item) for item in matches])

    async def search_menu(self, input_model: BaseModel) -> MenuItemsOutput:
        tool_input = _coerce(input_model, SearchMenuInput)
        items = await retrieval_search_menu(
            self.session,
            tool_input.tenant_id,
            tool_input.query,
            tool_input.restrictions.to_domain(),
            k=tool_input.k,
            embedding_provider=self.embedding_provider,
        )
        with span(
            "tool.search_menu",
            tool_name="search_menu",
            tenant_id=tool_input.tenant_id,
            retrieved_item_ids=[item.id for item in items],
        ):
            pass
        return MenuItemsOutput(items=[MenuItemViewSchema.from_domain(item) for item in items])

    async def dietary_filter(self, input_model: BaseModel) -> MenuItemsOutput:
        tool_input = _coerce(input_model, DietaryFilterInput)
        result = filter_safe_items(
            [item.to_domain() for item in tool_input.items],
            tool_input.restrictions.to_domain(),
        )
        with span(
            "tool.dietary_filter",
            tool_name="dietary_filter",
            retrieved_item_ids=[item.id for item in result.safe_items],
        ):
            pass
        return MenuItemsOutput(
            items=[MenuItemViewSchema.from_domain(item) for item in result.safe_items]
        )


def _coerce[T: BaseModel](input_model: BaseModel, model_type: type[T]) -> T:
    if isinstance(input_model, model_type):
        return input_model
    return model_type.model_validate(input_model)
