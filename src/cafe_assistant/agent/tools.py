"""Typed agent tool contracts for menu lookup, retrieval, and safety filtering.
Defines the Pydantic schemas passed between the state machine and deterministic tools.
These boundaries keep retrieval and dietary filtering explicit so the LLM never receives raw unsafe
menu data or decides safety itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.repositories.menu_repo import (
    load_menu_item_views_for_tenant,
    load_published_catalog_item_views_for_tenant,
)
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
    """Pydantic schema for restrictions schema data exchanged at an API or tool boundary."""
    avoid_allergens: set[AllergenCode] = Field(default_factory=set)
    modes: set[DietaryMode] = Field(default_factory=set)
    prefer_low_sugar: bool = False

    def to_domain(self) -> CustomerRestrictions:
        """Convert this schema object into the domain model used by deterministic safety code.

        Args:
            None.

        Returns:
            CustomerRestrictions:
                Domain object consumed by the safety filter or agent tool layer.
        """
        return CustomerRestrictions(
            avoid_allergens=set(self.avoid_allergens),
            modes=set(self.modes),
            prefer_low_sugar=self.prefer_low_sugar,
        )

    @classmethod
    def from_domain(cls, restrictions: CustomerRestrictions) -> RestrictionsSchema:
        """Build this schema object from a domain model for serialization.

        Args:
            restrictions (CustomerRestrictions):
                Customer allergen, dietary, and sugar preferences for the active turn.

        Returns:
            RestrictionsSchema:
                Schema object ready to serialize through a tool or API response.
        """
        return cls(
            avoid_allergens=set(restrictions.avoid_allergens),
            modes=set(restrictions.modes),
            prefer_low_sugar=restrictions.prefer_low_sugar,
        )


class MenuItemViewSchema(BaseModel):
    """Pydantic schema for menu item view schema data exchanged at an API or tool boundary."""
    id: int
    name: str
    allergen_codes: set[AllergenCode]
    dietary_tags: set[DietaryMode]
    allergen_data_complete: bool
    sugar_grams: float | None
    dietary_data_complete: bool = True

    @classmethod
    def from_domain(cls, item: MenuItemView) -> MenuItemViewSchema:
        """Build this schema object from a domain model for serialization.

        Args:
            item (MenuItemView):
                Menu or catalog item being transformed, embedded, or evaluated.

        Returns:
            MenuItemViewSchema:
                Schema object ready to serialize through a tool or API response.
        """
        return cls(
            id=item.id,
            name=item.name,
            allergen_codes=set(item.allergen_codes),
            dietary_tags=set(item.dietary_tags),
            allergen_data_complete=item.allergen_data_complete,
            sugar_grams=item.sugar_grams,
            dietary_data_complete=item.dietary_data_complete,
        )

    def to_domain(self) -> MenuItemView:
        """Convert this schema object into the domain model used by deterministic safety code.

        Args:
            None.

        Returns:
            MenuItemView:
                Domain object consumed by the safety filter or agent tool layer.
        """
        return MenuItemView(
            id=self.id,
            name=self.name,
            allergen_codes=set(self.allergen_codes),
            dietary_tags=set(self.dietary_tags),
            allergen_data_complete=self.allergen_data_complete,
            sugar_grams=self.sugar_grams,
            dietary_data_complete=self.dietary_data_complete,
        )


class MenuLookupInput(BaseModel):
    """Input schema for exact menu lookup through the agent tool boundary.

    The lookup tool is intentionally safety-aware: callers must provide the
    customer restrictions for the active turn so exact lookup cannot return raw
    menu candidates that bypass the deterministic dietary/allergen filter.
    """

    tenant_id: int
    query: str
    restrictions: RestrictionsSchema
    limit: int = 5


class SearchMenuInput(BaseModel):
    """Pydantic schema for search menu input data exchanged at an API or tool boundary."""
    tenant_id: int
    query: str
    restrictions: RestrictionsSchema
    k: int = 8


class DietaryFilterInput(BaseModel):
    """Pydantic schema for dietary filter input data exchanged at an API or tool boundary."""
    items: list[MenuItemViewSchema]
    restrictions: RestrictionsSchema


class MenuItemsOutput(BaseModel):
    """Output schema for menu tools that expose only safety-filtered items.

    The `items` field is the only menu content returned to callers and must
    contain safe `MenuItemViewSchema` records. `excluded_count` is metadata for
    control flow and observability; it does not expose unsafe menu item content.
    """

    items: list[MenuItemViewSchema]
    excluded_count: int = 0


ToolCallable = Callable[[BaseModel], Awaitable[BaseModel]]


class ToolRegistry:
    """Container for tool registry behavior and data."""
    def __init__(
        self,
        session: AsyncSession,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            session (AsyncSession):
                Async SQLAlchemy session used for tenant-scoped database reads and writes.
            embedding_provider (EmbeddingProvider | None):
                Embedding provider used to create query or record vectors.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        self.session = session
        self.embedding_provider = embedding_provider
        self._tools: dict[str, ToolCallable] = {
            "menu_lookup": self.menu_lookup,
            "search_menu": self.search_menu,
            "dietary_filter": self.dietary_filter,
        }

    async def call(self, name: str, input_model: BaseModel) -> BaseModel:
        """Handle call.

        Args:
            name (str):
                Name value required to perform this operation.
            input_model (BaseModel):
                Input model value required to perform this operation.

        Returns:
            BaseModel:
                Value produced for the caller according to the function contract.
        """
        with span("tool.call", tool_name=name):
            return await self._tools[name](input_model)

    async def menu_lookup(self, input_model: BaseModel) -> MenuItemsOutput:
        """Run exact menu lookup and return only safety-filtered menu items.

        Args:
            input_model (BaseModel):
                `MenuLookupInput` or compatible payload containing tenant,
                query, active restrictions, and result limit.

        Returns:
            MenuItemsOutput:
                Safe lookup matches plus a count of matching records excluded by
                the deterministic dietary/allergen filter.
        """
        tool_input = _coerce(input_model, MenuLookupInput)
        views = await load_published_catalog_item_views_for_tenant(
            self.session,
            tool_input.tenant_id,
        )
        if not views:
            views = await load_menu_item_views_for_tenant(self.session, tool_input.tenant_id)
        query = tool_input.query.lower().strip()
        raw_matches = [
            view
            for view in views
            if not query or query in view.name.lower() or view.name.lower() in query
        ]
        filter_result = filter_safe_items(
            raw_matches,
            tool_input.restrictions.to_domain(),
        )
        matches = filter_result.safe_items[: tool_input.limit]
        excluded_count = sum(1 for decision in filter_result.decisions if not decision.included)
        with span(
            "tool.menu_lookup",
            tool_name="menu_lookup",
            tenant_id=tool_input.tenant_id,
            retrieved_item_ids=[item.id for item in matches],
            excluded_item_count=excluded_count,
        ):
            pass
        return MenuItemsOutput(
            items=[MenuItemViewSchema.from_domain(item) for item in matches],
            excluded_count=excluded_count,
        )

    async def search_menu(self, input_model: BaseModel) -> MenuItemsOutput:
        """Retrieve menu candidates and return only items approved by the safety filter.

        Args:
            input_model (BaseModel):
                Input model value required to perform this operation.

        Returns:
            MenuItemsOutput:
                Safe menu item views after retrieval and deterministic filtering.
        """
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
        """Handle dietary filter.

        Args:
            input_model (BaseModel):
                Input model value required to perform this operation.

        Returns:
            MenuItemsOutput:
                Value produced for the caller according to the function contract.
        """
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
    """Handle coerce.

    Args:
        input_model (BaseModel):
            Input model value required to perform this operation.
        model_type (type[T]):
            Model type value required to perform this operation.

    Returns:
        T:
            Value produced for the caller according to the function contract.
    """
    if isinstance(input_model, model_type):
        return input_model
    return model_type.model_validate(input_model)
