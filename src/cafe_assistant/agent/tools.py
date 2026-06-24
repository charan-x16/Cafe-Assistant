"""Typed agent tool contracts for menu lookup, retrieval, and safety filtering.

The state machine calls these tools instead of reaching directly into retrieval
or database code. Each input and output is a Pydantic model so tenant scope,
customer restrictions, and safe menu results stay explicit at every boundary.
Menu lookup and retrieval both run deterministic safety filtering before
returning items, which preserves the invariant that composition never sees raw
unsafe menu candidates.
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
    """Serializable customer restrictions passed into deterministic tools.

    The domain filter owns the safety semantics. This schema only carries those
    values across API/tool boundaries without weakening allergen or dietary
    guarantees.
    """

    avoid_allergens: set[AllergenCode] = Field(default_factory=set)
    modes: set[DietaryMode] = Field(default_factory=set)
    prefer_low_sugar: bool = False

    def to_domain(self) -> CustomerRestrictions:
        """Convert this schema object into the domain model used by deterministic safety code.

        Args:
            None:
                This method reads the fields stored on the schema instance.

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
    """Serializable safe menu item view returned by agent tools.

    The fields mirror `MenuItemView` so tool callers can pass filtered menu
    candidates between retrieval, fallback filtering, and composition without
    exposing ORM objects or raw rows.
    """

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
            None:
                This method reads the fields stored on the schema instance.

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
    menu candidates that bypass the deterministic dietary/allergen filter. The
    optional limit is set to None only by trusted internal browse paths that
    need the complete safe catalog rather than a short exact-match result.
    """

    tenant_id: int
    query: str
    restrictions: RestrictionsSchema
    limit: int | None = 5


class SearchMenuInput(BaseModel):
    """Input schema for hybrid menu search through the agent tool boundary.

    Search receives the active tenant, user query, restrictions, and result
    size. The tool returns only post-filter safe menu item views.
    """

    tenant_id: int
    query: str
    restrictions: RestrictionsSchema
    k: int = 8


class DietaryFilterInput(BaseModel):
    """Input schema for rerunning deterministic safety filtering on known items.

    This is used by fallback paths that already have item candidates but must
    still enforce the allergen/dietary gate before anything reaches the model.
    """

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
    """Registry of deterministic tools available to the chat state machine.

    The registry centralizes tool names, input validation, tenant-scoped session
    access, and optional embedding provider injection. The LLM never calls this
    object directly; the state machine invokes named tools and enforces budgets.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Create the tool registry for one request-scoped database session.

        Args:
            session (AsyncSession):
                Async SQLAlchemy session used for tenant-scoped database reads and writes.
            embedding_provider (EmbeddingProvider | None):
                Optional provider used by hybrid retrieval to embed the query.
                Tests inject deterministic providers here so no network calls are required.

        Returns:
            None:
                The registry stores dependencies and named tool callables for later use.
        """
        self.session = session
        self.embedding_provider = embedding_provider
        self._tools: dict[str, ToolCallable] = {
            "menu_lookup": self.menu_lookup,
            "search_menu": self.search_menu,
            "dietary_filter": self.dietary_filter,
        }

    async def call(self, name: str, input_model: BaseModel) -> BaseModel:
        """Dispatch one named tool invocation.

        Args:
            name (str):
                Registered tool name. Supported names are `menu_lookup`,
                `search_menu`, and `dietary_filter`.
            input_model (BaseModel):
                Pydantic input object expected by the selected tool.

        Returns:
            BaseModel:
                Pydantic output object produced by the selected tool.
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
        if tool_input.limit is None:
            matches = filter_result.safe_items
        else:
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
                `SearchMenuInput` or compatible payload containing tenant,
                query, active restrictions, and result count.

        Returns:
            MenuItemsOutput:
                Safe menu item views after hybrid retrieval and deterministic filtering.
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
        """Apply the deterministic dietary/allergen filter to supplied item views.

        Args:
            input_model (BaseModel):
                `DietaryFilterInput` or compatible payload containing candidate
                item views and active customer restrictions.

        Returns:
            MenuItemsOutput:
                Only the supplied items that pass the deterministic filter.
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
    """Validate or reuse a Pydantic tool input object.

    Args:
        input_model (BaseModel):
            Existing Pydantic object passed through the generic registry boundary.
        model_type (type[T]):
            Concrete schema class expected by the destination tool.

    Returns:
        T:
            `input_model` when it is already the expected type, otherwise a
            validated instance of `model_type`.
    """
    if isinstance(input_model, model_type):
        return input_model
    return model_type.model_validate(input_model)
