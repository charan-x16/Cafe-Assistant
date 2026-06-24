"""Explicit chat-agent state machine for safe menu recommendations.

The state machine routes each request, carries session/profile restrictions,
invokes deterministic menu tools, and passes only safety-filtered menu items to
response composition. It enforces the rule that LLM synthesis never sees raw
menu candidates or decides allergen/dietary safety.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.agent.composer import ComposeInput, ResponseComposer
from cafe_assistant.agent.restrictions import extract_restrictions
from cafe_assistant.agent.router import Intent, MessageRouter
from cafe_assistant.agent.tools import (
    DietaryFilterInput,
    MenuItemsOutput,
    MenuItemViewSchema,
    MenuLookupInput,
    RestrictionsSchema,
    SearchMenuInput,
    ToolRegistry,
)
from cafe_assistant.config import settings
from cafe_assistant.db.repositories.menu_repo import (
    MenuBrowseItem,
    load_menu_browse_items_for_tenant,
)
from cafe_assistant.db.repositories.profile_repo import get_customer
from cafe_assistant.domain.dietary import CustomerRestrictions, MenuItemView, filter_safe_items
from cafe_assistant.gateway.model_gateway import (
    ChatMessage,
    ChatModelCascade,
    EmbeddingProvider,
    get_chat_model_cascade,
)
from cafe_assistant.identity.device import verify_device_token
from cafe_assistant.memory.profile import load_durable_profile, merge_profile_with_session
from cafe_assistant.memory.session import (
    SessionMemory,
    SessionState,
    append_turns,
    get_redis_session_memory,
)
from cafe_assistant.memory.write_gate import (
    classify_candidate_writes,
    extract_preferences,
    persist_allowed_writes,
)
from cafe_assistant.observability.metrics import record_quality_event
from cafe_assistant.observability.tracing import finish_trace, span, start_trace
from cafe_assistant.security.audit import AuditContext, append_audit_event

T = TypeVar("T")
SAFE_FAILURE_RESPONSE = "Sorry, I could not complete that safely right now."


class AgentState(StrEnum):
    """Ordered orchestration states recorded for every chat-agent run."""

    CLASSIFIED = "CLASSIFIED"
    RETRIEVING = "RETRIEVING"
    FILTERING = "FILTERING"
    RECOMMENDING = "RECOMMENDING"
    COMPOSING = "COMPOSING"
    COMPLETE = "COMPLETE"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ChatAgentRequest:
    """Request envelope passed into the chat agent.

    The envelope carries tenant scope, session identity, optional durable
    customer identity, and trace metadata. User text remains untrusted data and
    is never treated as instructions for the orchestrator itself.
    """

    session_id: str
    tenant_id: int
    message: str
    device_token: str | None = None
    customer_id: int | None = None
    location_id: int | None = None
    table_id: str | None = None
    request_id: str = "internal"
    trace_id: str = "internal"
    actor: str = "anonymous"


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Runtime limits for one agent invocation.

    max_tool_calls bounds deterministic tool usage, deadline_seconds bounds
    router/tool/composer work, and search_k controls recommendation candidates.
    """

    max_tool_calls: int = settings.agent_max_tool_calls
    deadline_seconds: float = settings.agent_deadline_seconds
    search_k: int = 8


@dataclass(slots=True)
class ChatAgentResult:
    """Final non-streaming result returned by the chat agent.

    The response is customer-facing text. `safe_items` contains only menu items
    that passed deterministic filtering, and `model_messages` is retained for
    tests/trace inspection of the guarded composition context.
    """

    response: str
    state_history: list[AgentState]
    safe_items: list[MenuItemView] = field(default_factory=list)
    restrictions: CustomerRestrictions = field(
        default_factory=lambda: CustomerRestrictions(
            avoid_allergens=set(),
            modes=set(),
            prefer_low_sugar=False,
        )
    )
    model_messages: list[ChatMessage] = field(default_factory=list)
    tool_calls: int = 0
    customer_id: int | None = None


@dataclass(slots=True)
class _PreparedResponse:
    """Intermediate orchestration result before final composition.

    A non-None `response` means the state machine has already produced a safe
    fallback, refusal, or smalltalk response. Otherwise the caller may compose
    using the included safe item set only.
    """

    response: str | None
    safe_items: list[MenuItemView]
    restrictions: CustomerRestrictions
    current_turn_restrictions: CustomerRestrictions
    current_turn_preferences: dict[str, object]
    preferences: dict[str, object]
    state_history: list[AgentState]
    medical_disclaimer: bool
    tool_calls: int
    customer_id: int | None


@dataclass(slots=True)
class _RunControls:
    """Mutable per-request controls for deadline and tool-budget enforcement.

    The same object is passed through router, retrieval, and fallback paths so
    every tool call is counted against one budget and every await uses the same
    absolute deadline.
    """

    deadline_at: float
    tool_calls: int = 0


class ToolBudgetExceededError(RuntimeError):
    """Raised when an agent run attempts more tool calls than allowed."""

    pass


class RequestDeadlineExceededError(RuntimeError):
    """Raised when router, tool, or composer work exceeds the request deadline."""

    pass


class RecommenderUnavailableError(RuntimeError):
    """Raised when fallback retrieval returns an invalid tool result."""

    pass


class ChatAgent:
    """Single explicit state machine for safe cafe menu chat.

    The agent owns routing, memory merge, deterministic retrieval/filtering,
    fallback handling, auditing, and composition. It never lets the LLM choose
    safety status or see raw unsafe menu candidates.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        memory: SessionMemory | None = None,
        chat_models: ChatModelCascade | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """Create a request orchestrator with injectable infrastructure.

        Args:
            session (AsyncSession):
                Async database session used by tenant-scoped repositories and tools.
            memory (SessionMemory | None):
                Optional session memory implementation. Tests pass in-memory
                storage; production defaults to Redis-backed memory.
            chat_models (ChatModelCascade | None):
                Optional cheap/strong chat provider cascade used by routing and composition.
            embedding_provider (EmbeddingProvider | None):
                Optional embedding provider used by retrieval tools.
            config (AgentConfig | None):
                Optional runtime limits for deadline, tool budget, and candidate count.

        Returns:
            None:
                The agent stores dependencies and creates router, tool registry,
                and composer collaborators for later runs.
        """
        self.session = session
        self.memory = memory or get_redis_session_memory()
        self.chat_models = chat_models or get_chat_model_cascade()
        self.embedding_provider = embedding_provider
        self.config = config or AgentConfig()
        self.router = MessageRouter(self.chat_models.cheap, self.chat_models.strong)
        self.tools = ToolRegistry(session, embedding_provider=embedding_provider)
        self.composer = ResponseComposer(self.chat_models.strong)

    async def run(self, request: ChatAgentRequest) -> ChatAgentResult:
        """Run the non-streaming chat state machine for one request.

        Args:
            request (ChatAgentRequest):
                Tenant-scoped chat request containing user text, session ID, and trace IDs.

        Returns:
            ChatAgentResult:
                Customer-facing response plus state history, safe item context,
                restrictions, and diagnostic metadata. Any unhandled exception is
                converted to a generic safe failure response.
        """
        start_trace(
            tenant_id=request.tenant_id,
            request_id=request.request_id,
            trace_id=request.trace_id,
        )
        try:
            with span(
                "agent.run",
                tenant_id=request.tenant_id,
                request_id=request.request_id,
            ):
                deadline_at = asyncio.get_running_loop().time() + self.config.deadline_seconds
                prepared = await self._prepare_response(request, deadline_at=deadline_at)
                if prepared.response is not None:
                    await self._save_turn(
                        request,
                        prepared.restrictions,
                        prepared.response,
                        customer_id=prepared.customer_id,
                        current_turn_restrictions=prepared.current_turn_restrictions,
                        current_turn_preferences=prepared.current_turn_preferences,
                    )
                    await self._audit_recommendation(request, prepared)
                    return ChatAgentResult(
                        response=prepared.response,
                        state_history=prepared.state_history,
                        safe_items=prepared.safe_items,
                        restrictions=prepared.restrictions,
                        tool_calls=prepared.tool_calls,
                        customer_id=prepared.customer_id,
                    )

                response = await self.composer.compose(
                    ComposeInput(
                        user_message=request.message,
                        safe_items=prepared.safe_items,
                        restrictions=prepared.restrictions,
                        preferences=prepared.preferences,
                        include_medical_disclaimer=prepared.medical_disclaimer,
                    ),
                    timeout_seconds=self._remaining_seconds(deadline_at),
                )
                state_history = [*prepared.state_history, AgentState.COMPLETE]
                await self._save_turn(
                    request,
                    prepared.restrictions,
                    response,
                    customer_id=prepared.customer_id,
                    current_turn_restrictions=prepared.current_turn_restrictions,
                    current_turn_preferences=prepared.current_turn_preferences,
                )
                await self._audit_recommendation(request, prepared)
                return ChatAgentResult(
                    response=response,
                    state_history=state_history,
                    safe_items=prepared.safe_items,
                    restrictions=prepared.restrictions,
                    model_messages=list(self.composer.last_messages),
                    tool_calls=prepared.tool_calls,
                    customer_id=prepared.customer_id,
                )
        except Exception:  # noqa: BLE001 - user-facing fallback must catch all failures.
            record_quality_event("errors_agent_total")
            return ChatAgentResult(
                response=SAFE_FAILURE_RESPONSE,
                state_history=[AgentState.FAILED],
            )
        finally:
            finish_trace(request.trace_id)

    async def stream_response(self, request: ChatAgentRequest) -> AsyncIterator[str]:
        """Stream the chat response while preserving the same safety boundary.

        Args:
            request (ChatAgentRequest):
                Tenant-scoped chat request containing user text, session ID, and trace IDs.

        Returns:
            AsyncIterator[str]:
                Response chunks. On unexpected failure the stream yields the same
                generic safe failure text used by `run`.
        """
        try:
            deadline_at = asyncio.get_running_loop().time() + self.config.deadline_seconds
            prepared = await self._prepare_response(request, deadline_at=deadline_at)
            if prepared.response is not None:
                for chunk in _chunk_text(prepared.response):
                    yield chunk
                await self._save_turn(
                    request,
                    prepared.restrictions,
                    prepared.response,
                    customer_id=prepared.customer_id,
                    current_turn_restrictions=prepared.current_turn_restrictions,
                    current_turn_preferences=prepared.current_turn_preferences,
                )
                await self._audit_recommendation(request, prepared)
                return

            chunks: list[str] = []
            async for token in self.composer.stream(
                ComposeInput(
                    user_message=request.message,
                    safe_items=prepared.safe_items,
                    restrictions=prepared.restrictions,
                    preferences=prepared.preferences,
                    include_medical_disclaimer=prepared.medical_disclaimer,
                ),
                timeout_seconds=self._remaining_seconds(deadline_at),
            ):
                chunks.append(token)
                yield token
            await self._save_turn(
                request,
                prepared.restrictions,
                "".join(chunks),
                customer_id=prepared.customer_id,
                current_turn_restrictions=prepared.current_turn_restrictions,
                current_turn_preferences=prepared.current_turn_preferences,
            )
            await self._audit_recommendation(request, prepared)
        except Exception:
            yield SAFE_FAILURE_RESPONSE

    async def _prepare_response(
        self,
        request: ChatAgentRequest,
        *,
        deadline_at: float,
    ) -> _PreparedResponse:
        """Prepare routing, restrictions, retrieval, and fallback state.

        Args:
            request (ChatAgentRequest):
                Tenant-scoped chat request being processed.
            deadline_at (float):
                Absolute event-loop timestamp by which router/tool work must finish.

        Returns:
            _PreparedResponse:
                Either an immediate safe response or the safe item context needed
                by the composer.
        """
        state_history: list[AgentState] = []
        controls = _RunControls(deadline_at=deadline_at)

        try:
            session_state = await self.memory.load(
                tenant_id=request.tenant_id,
                session_id=request.session_id,
            )
        except Exception:
            record_quality_event("memory_unavailable_total")
            session_state = SessionState()
        customer_id = await self._resolve_customer_id(request)
        durable_profile = await load_durable_profile(
            self.session,
            tenant_id=request.tenant_id,
            customer_id=customer_id,
        )
        memory_context = merge_profile_with_session(
            session_state=session_state,
            durable_profile=durable_profile,
        )
        extraction = extract_restrictions(
            request.message,
            memory_context.session_state.restrictions,
        )
        current_turn_preferences = extract_preferences(request.message)
        active_preferences = {
            **memory_context.preferences,
            **current_turn_preferences,
        }
        search_query = _query_with_preferences(request.message, active_preferences)
        broad_menu_request = _is_broad_menu_request(request.message)

        # Simple deterministic menu browsing and medical refusals run before routing.
        if extraction.medical_question:
            response = (
                "I can't help with insulin, carb-counting, or other medical decisions. "
                "This is not medical advice; please check with a clinician or cafe staff."
            )
            record_quality_event("medical_refusals_total")
            return _PreparedResponse(
                response=response,
                safe_items=[],
                restrictions=extraction.restrictions,
                current_turn_restrictions=extraction.current_turn_restrictions,
                current_turn_preferences=current_turn_preferences,
                preferences=active_preferences,
                state_history=[*state_history, AgentState.CLASSIFIED, AgentState.ESCALATED],
                medical_disclaimer=True,
                tool_calls=controls.tool_calls,
                customer_id=customer_id,
            )

        possible_section_request = _might_be_menu_section_request(request.message)
        if broad_menu_request or possible_section_request:
            browse_entries = await self._safe_menu_browse_items(
                request,
                extraction.restrictions,
                controls,
            )
            matched_section = _match_menu_section(browse_entries, request.message)
            if broad_menu_request or matched_section is not None:
                if not browse_entries:
                    response = (
                        "I can't confirm a safe menu option from the available menu data. "
                        "Please check with cafe staff before ordering."
                    )
                    record_quality_event("empty_safe_sets_total", reason="menu_browse_empty")
                    return _PreparedResponse(
                        response=response,
                        safe_items=[],
                        restrictions=extraction.restrictions,
                        current_turn_restrictions=extraction.current_turn_restrictions,
                        current_turn_preferences=current_turn_preferences,
                        preferences=active_preferences,
                        state_history=[
                            *state_history,
                            AgentState.RETRIEVING,
                            AgentState.FILTERING,
                            AgentState.COMPLETE,
                        ],
                        medical_disclaimer=False,
                        tool_calls=controls.tool_calls,
                        customer_id=customer_id,
                    )
                if matched_section is None:
                    response = _format_menu_sections_response(
                        browse_entries,
                        extraction.restrictions,
                    )
                    safe_items = [entry.item for entry in browse_entries]
                else:
                    response = _format_menu_section_items_response(
                        matched_section,
                        extraction.restrictions,
                    )
                    safe_items = [entry.item for entry in matched_section.entries]
                return _PreparedResponse(
                    response=response,
                    safe_items=safe_items,
                    restrictions=extraction.restrictions,
                    current_turn_restrictions=extraction.current_turn_restrictions,
                    current_turn_preferences=current_turn_preferences,
                    preferences=active_preferences,
                    state_history=[
                        *state_history,
                        AgentState.RETRIEVING,
                        AgentState.FILTERING,
                        AgentState.RECOMMENDING,
                        AgentState.COMPOSING,
                        AgentState.COMPLETE,
                    ],
                    medical_disclaimer=False,
                    tool_calls=controls.tool_calls,
                    customer_id=customer_id,
                )
        self._ensure_deadline(deadline_at)
        classification = await self._run_with_deadline(
            self.router.classify(request.message),
            controls,
        )
        state_history.append(AgentState.CLASSIFIED)

        if classification.intent in {Intent.OUT_OF_SCOPE, Intent.SMALLTALK}:
            response = _non_menu_response(classification.intent)
            return _PreparedResponse(
                response=response,
                safe_items=[],
                restrictions=extraction.restrictions,
                current_turn_restrictions=extraction.current_turn_restrictions,
                current_turn_preferences=current_turn_preferences,
                preferences=active_preferences,
                state_history=[*state_history, AgentState.COMPLETE],
                medical_disclaimer=False,
                tool_calls=controls.tool_calls,
                customer_id=customer_id,
            )

        if classification.intent == Intent.DIETARY_SAFETY:
            response = _format_restriction_acknowledgement(
                extraction.current_turn_restrictions,
            )
            return _PreparedResponse(
                response=response,
                safe_items=[],
                restrictions=extraction.restrictions,
                current_turn_restrictions=extraction.current_turn_restrictions,
                current_turn_preferences=current_turn_preferences,
                preferences=active_preferences,
                state_history=[*state_history, AgentState.COMPLETE],
                medical_disclaimer=False,
                tool_calls=controls.tool_calls,
                customer_id=customer_id,
            )

        self._ensure_deadline(deadline_at)
        state_history.append(AgentState.RETRIEVING)
        self._charge_tool_call(controls)
        lookup_output = await self._safe_menu_lookup(
            request,
            extraction.restrictions,
            controls,
        )
        if lookup_output.excluded_count > 0 and not lookup_output.items:
            response = (
                "I can't confirm a safe option for that request based on your restrictions. "
                "Please check with cafe staff before ordering."
            )
            record_quality_event("empty_safe_sets_total", reason="lookup_excluded_all_matches")
            return _PreparedResponse(
                response=response,
                safe_items=[],
                restrictions=extraction.restrictions,
                current_turn_restrictions=extraction.current_turn_restrictions,
                current_turn_preferences=current_turn_preferences,
                preferences=active_preferences,
                state_history=[*state_history, AgentState.FILTERING, AgentState.COMPLETE],
                medical_disclaimer=False,
                tool_calls=controls.tool_calls,
                customer_id=customer_id,
            )

        self._charge_tool_call(controls)
        output = await self._safe_search_menu(
            request,
            query=search_query,
            restrictions=extraction.restrictions,
            controls=controls,
        )

        self._ensure_deadline(deadline_at)
        state_history.append(AgentState.FILTERING)
        safe_items = [item.to_domain() for item in output.items]
        with span(
            "agent.filtering",
            retrieved_item_ids=[item.id for item in safe_items],
            safe_item_ids=[item.id for item in safe_items],
        ):
            pass
        if not safe_items:
            try:
                fallback_output = await self._fallback_popular_items(
                    request,
                    extraction.restrictions,
                    controls,
                    result_limit=self.config.search_k,
                )
                safe_items = [item.to_domain() for item in fallback_output.items]
                if safe_items:
                    record_quality_event("recommender_fallback_total", stage="empty_search")
            except (RequestDeadlineExceededError, ToolBudgetExceededError):
                raise
            except Exception:
                record_quality_event("recommender_fallback_total", stage="empty_search_failed")

        if not safe_items:
            response = (
                "I can't confirm a safe menu option from the available menu data. "
                "Please check with cafe staff before ordering."
            )
            record_quality_event("empty_safe_sets_total", reason="filter_result_empty")
            return _PreparedResponse(
                response=response,
                safe_items=[],
                restrictions=extraction.restrictions,
                current_turn_restrictions=extraction.current_turn_restrictions,
                current_turn_preferences=current_turn_preferences,
                preferences=active_preferences,
                state_history=[*state_history, AgentState.COMPLETE],
                medical_disclaimer=False,
                tool_calls=controls.tool_calls,
                customer_id=customer_id,
            )

        state_history.append(AgentState.RECOMMENDING)
        recommended_items = safe_items[:3]
        state_history.append(AgentState.COMPOSING)
        return _PreparedResponse(
            response=None,
            safe_items=recommended_items,
            restrictions=extraction.restrictions,
            current_turn_restrictions=extraction.current_turn_restrictions,
            current_turn_preferences=current_turn_preferences,
            preferences=active_preferences,
            state_history=state_history,
            medical_disclaimer=False,
            tool_calls=controls.tool_calls,
            customer_id=customer_id,
        )

    async def _safe_menu_browse_items(
        self,
        request: ChatAgentRequest,
        restrictions: CustomerRestrictions,
        controls: _RunControls,
    ) -> list[MenuBrowseItem]:
        """Load all browseable menu entries and apply the deterministic safety filter.

        Args:
            request (ChatAgentRequest):
                Chat request containing the tenant scope for menu browsing.
            restrictions (CustomerRestrictions):
                Active restrictions that must be applied before item names are shown.
            controls (_RunControls):
                Request controls carrying the deadline and tool-call counter.

        Returns:
            list[MenuBrowseItem]:
                Ordered browse entries whose `item` values passed the safety filter.
        """
        self._charge_tool_call(controls)
        try:
            entries = await self._run_with_deadline(
                load_menu_browse_items_for_tenant(self.session, request.tenant_id),
                controls,
            )
        except (RequestDeadlineExceededError, ToolBudgetExceededError):
            raise
        except Exception:
            record_quality_event("recommender_fallback_total", stage="menu_browse")
            return []
        filter_result = filter_safe_items(
            [entry.item for entry in entries],
            restrictions,
        )
        safe_ids = {item.id for item in filter_result.safe_items}
        with span(
            "agent.menu_browse_filter",
            retrieved_item_ids=[entry.item.id for entry in entries],
            safe_item_ids=[item.id for item in filter_result.safe_items],
        ):
            pass
        return [entry for entry in entries if entry.item.id in safe_ids]
    async def _safe_menu_lookup(
        self,
        request: ChatAgentRequest,
        restrictions: CustomerRestrictions,
        controls: _RunControls,
    ) -> MenuItemsOutput:
        """Run exact lookup with deadline and active safety restrictions.

        Args:
            request (ChatAgentRequest):
                Chat request containing tenant, session, and user message data.
            restrictions (CustomerRestrictions):
                Customer restrictions extracted for the current turn and session.
            controls (_RunControls):
                Request controls carrying the deadline and tool-call counter.

        Returns:
            MenuItemsOutput:
                Safety-filtered lookup results, or an empty output if lookup fails
                for a non-control reason.
        """
        try:
            output = await self._run_with_deadline(
                self.tools.call(
                    "menu_lookup",
                    MenuLookupInput(
                        tenant_id=request.tenant_id,
                        query=request.message,
                        restrictions=RestrictionsSchema.from_domain(restrictions),
                        limit=3,
                    ),
                ),
                controls,
            )
            if not isinstance(output, MenuItemsOutput):
                raise TypeError("menu_lookup tool returned an unexpected output type.")
            return output
        except (RequestDeadlineExceededError, ToolBudgetExceededError):
            raise
        except Exception:
            record_quality_event("recommender_fallback_total", stage="menu_lookup")
            return MenuItemsOutput(items=[])

    async def _safe_search_menu(
        self,
        request: ChatAgentRequest,
        *,
        query: str,
        restrictions: CustomerRestrictions,
        controls: _RunControls,
    ) -> MenuItemsOutput:
        """Run hybrid search and fall back to safe popular items on failure.

        Args:
            request (ChatAgentRequest):
                Chat request containing tenant and trace context.
            query (str):
                Search query augmented with durable preferences when present.
            restrictions (CustomerRestrictions):
                Customer restrictions extracted for the current turn and session.
            controls (_RunControls):
                Request controls carrying the deadline and tool-call counter.

        Returns:
            MenuItemsOutput:
                Safety-filtered search results, or safety-filtered fallback items
                when the primary recommender fails before the deadline.
        """
        try:
            output = await self._run_with_deadline(
                self.tools.call(
                    "search_menu",
                    SearchMenuInput(
                        tenant_id=request.tenant_id,
                        query=query,
                        restrictions=RestrictionsSchema.from_domain(restrictions),
                        k=self.config.search_k,
                    ),
                ),
                controls,
            )
            if not isinstance(output, MenuItemsOutput):
                raise TypeError("search_menu tool returned an unexpected output type.")
            return output
        except (RequestDeadlineExceededError, ToolBudgetExceededError):
            raise
        except Exception:
            record_quality_event("recommender_fallback_total", stage="search_menu")
            return await self._fallback_popular_items(
                request, restrictions, controls, result_limit=self.config.search_k
            )

    async def _fallback_popular_items(
        self,
        request: ChatAgentRequest,
        restrictions: CustomerRestrictions,
        controls: _RunControls,
        *,
        result_limit: int | None = None,
    ) -> MenuItemsOutput:
        """Load fallback or broad-browse items while respecting budget and deadline.

        Args:
            request (ChatAgentRequest):
                Chat request that supplies the tenant scope for fallback lookup.
            restrictions (CustomerRestrictions):
                Customer restrictions that must still be enforced during fallback.
            controls (_RunControls):
                Request controls carrying the deadline and tool-call counter.
            result_limit (int | None):
                Optional safe result cap. None returns every item that passed
                the deterministic filter, which is used for broad menu browsing.

        Returns:
            MenuItemsOutput:
                Safety-filtered items, either complete for broad browsing or capped for fallback.
        """
        self._charge_tool_call(controls)
        all_items = await self._run_with_deadline(
            self.tools.call(
                "menu_lookup",
                MenuLookupInput(
                    tenant_id=request.tenant_id,
                    query="",
                    restrictions=RestrictionsSchema.from_domain(restrictions),
                    limit=None,
                ),
            ),
            controls,
        )
        if not isinstance(all_items, MenuItemsOutput):
            raise RecommenderUnavailableError("Fallback menu lookup failed.")
        self._charge_tool_call(controls)
        result = await self._run_with_deadline(
            self.tools.call(
                "dietary_filter",
                DietaryFilterInput(
                    items=[
                        MenuItemViewSchema.from_domain(item.to_domain())
                        for item in all_items.items
                    ],
                    restrictions=RestrictionsSchema.from_domain(restrictions),
                ),
            ),
            controls,
        )
        if not isinstance(result, MenuItemsOutput):
            raise RecommenderUnavailableError("Fallback dietary filter failed.")
        if result_limit is None:
            return result
        return MenuItemsOutput(items=result.items[:result_limit])

    async def _resolve_customer_id(self, request: ChatAgentRequest) -> int | None:
        """Resolve an optional durable customer identity within the request tenant.

        Args:
            request (ChatAgentRequest):
                Chat request containing either an explicit customer ID or a device token.

        Returns:
            int | None:
                Customer ID when it exists for the same tenant, otherwise None so
                the run continues as an anonymous session.
        """
        if request.customer_id is not None:
            customer = await get_customer(
                self.session,
                tenant_id=request.tenant_id,
                customer_id=request.customer_id,
            )
            return customer.id if customer is not None else None
        identity = await verify_device_token(
            self.session,
            tenant_id=request.tenant_id,
            token=request.device_token,
        )
        return identity.customer_id if identity is not None else None

    async def _save_turn(
        self,
        request: ChatAgentRequest,
        restrictions: CustomerRestrictions,
        response: str,
        *,
        customer_id: int | None,
        current_turn_restrictions: CustomerRestrictions,
        current_turn_preferences: dict[str, object],
    ) -> None:
        """Persist recent session turns and allowed durable memory writes.

        Args:
            request (ChatAgentRequest):
                Chat request containing the session key, tenant, and user message.
            restrictions (CustomerRestrictions):
                Active restrictions extracted for the current turn.
            response (str):
                Assistant response text to append to session memory.
            customer_id (int | None):
                Durable customer ID when the session is recognized and tenant-scoped.
            current_turn_restrictions (CustomerRestrictions):
                Positive health/dietary facts explicitly stated in the current turn.
            current_turn_preferences (dict[str, object]):
                Non-health preferences explicitly stated in the current turn.

        Returns:
            None:
                Session memory and permitted profile writes are updated when
                backing stores are available.
        """
        try:
            current_state = await self.memory.load(
                tenant_id=request.tenant_id,
                session_id=request.session_id,
            )
        except Exception:
            current_state = SessionState()
        state = SessionState(
            restrictions=restrictions,
            preferences={
                **current_state.preferences,
                **current_turn_preferences,
            },
            recent_turns=current_state.recent_turns,
        )
        try:
            await self.memory.save(
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                state=append_turns(state, request.message, response),
            )
        except Exception:
            record_quality_event("memory_unavailable_total")
        if customer_id is None:
            return

        writes = classify_candidate_writes(
            message=request.message,
            current_turn_restrictions=current_turn_restrictions,
            current_turn_preferences=current_turn_preferences,
        )
        if writes:
            result = await persist_allowed_writes(
                self.session,
                tenant_id=request.tenant_id,
                customer_id=customer_id,
                writes=writes,
            )
            if result.persisted:
                await append_audit_event(
                    self.session,
                    context=self._audit_context(request, customer_id=customer_id),
                    action="profile_write",
                    payload={
                        "customer_id": customer_id,
                        "persisted_kinds": [write.kind.value for write in result.persisted],
                        "skipped_kinds": [write.kind.value for write in result.skipped],
                        "location_id": request.location_id,
                        "table_id": request.table_id,
                    },
                )

    async def _audit_recommendation(
        self,
        request: ChatAgentRequest,
        prepared: _PreparedResponse,
    ) -> None:
        """Append an audit event when a recommendation reaches the customer.

        Args:
            request (ChatAgentRequest):
                Chat request carrying tenant, request, and trace identifiers.
            prepared (_PreparedResponse):
                Prepared response containing the final safe item set and customer identity.

        Returns:
            None:
                An audit row is written only when there are recommended safe items.
        """
        if not prepared.safe_items:
            return
        await append_audit_event(
            self.session,
            context=self._audit_context(request, customer_id=prepared.customer_id),
            action="recommendation_served",
            payload={
                "session_id": request.session_id,
                "item_ids": [item.id for item in prepared.safe_items],
                "item_names": [item.name for item in prepared.safe_items],
                "customer_id": prepared.customer_id,
                "location_id": request.location_id,
                "table_id": request.table_id,
            },
        )

    def _audit_context(
        self,
        request: ChatAgentRequest,
        *,
        customer_id: int | None,
    ) -> AuditContext:
        """Build tenant-scoped audit context for profile and recommendation events.

        Args:
            request (ChatAgentRequest):
                Chat request containing tenant, request, trace, and actor metadata.
            customer_id (int | None):
                Durable customer ID when the current run is recognized.

        Returns:
            AuditContext:
                Context object passed to append-only audit logging.
        """
        return AuditContext(
            tenant_id=request.tenant_id,
            actor=f"customer:{customer_id}" if customer_id is not None else request.actor,
            request_id=request.request_id,
            trace_id=request.trace_id,
        )

    def _charge_tool_call(self, controls: _RunControls) -> None:
        """Count one planned tool call and enforce the configured budget.

        Args:
            controls (_RunControls):
                Mutable per-request controls holding the current tool-call count.

        Returns:
            None:
                The counter is incremented in place. Exceeding the budget raises
                `ToolBudgetExceededError` before the tool is invoked.
        """
        controls.tool_calls += 1
        self._ensure_tool_budget(controls.tool_calls)

    async def _run_with_deadline(self, awaitable: Awaitable[T], controls: _RunControls) -> T:
        """Await router or tool work using the remaining request deadline.

        Args:
            awaitable (Awaitable[T]):
                Coroutine or awaitable operation to run under the deadline.
            controls (_RunControls):
                Per-request controls containing the absolute deadline timestamp.

        Returns:
            T:
                Result returned by the awaited router or tool operation before the deadline expires.
        """
        self._ensure_deadline(controls.deadline_at)
        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=self._remaining_seconds(controls.deadline_at),
            )
        except TimeoutError as exc:
            raise RequestDeadlineExceededError("Agent request deadline exceeded.") from exc

    def _ensure_tool_budget(self, tool_calls: int) -> None:
        """Validate the current tool-call count against the agent configuration.

        Args:
            tool_calls (int):
                Number of tool calls charged so far in the current request.

        Returns:
            None:
                The function returns normally when the count is within budget.
        """
        if tool_calls > self.config.max_tool_calls:
            raise ToolBudgetExceededError("Agent tool-call budget exceeded.")

    def _ensure_deadline(self, deadline_at: float) -> None:
        """Validate that the request deadline has not already expired.

        Args:
            deadline_at (float):
                Absolute event-loop timestamp for the end of the request budget.

        Returns:
            None:
                The function returns normally when there is still time remaining.
        """
        if asyncio.get_running_loop().time() > deadline_at:
            raise RequestDeadlineExceededError("Agent request deadline exceeded.")

    def _remaining_seconds(self, deadline_at: float) -> float:
        """Calculate remaining request time for a bounded await.

        Args:
            deadline_at (float):
                Absolute event-loop timestamp for the end of the request budget.

        Returns:
            float:
                Positive number of seconds remaining before the request deadline.
        """
        remaining = deadline_at - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RequestDeadlineExceededError("Agent request deadline exceeded.")
        return remaining


def _non_menu_response(intent: Intent) -> str:
    """Return deterministic copy for routes that should not retrieve the menu.

    Args:
        intent (Intent):
            Router intent classified as smalltalk or out-of-scope.

    Returns:
        str:
            Short customer-facing response that does not call retrieval or composition.
    """
    if intent == Intent.SMALLTALK:
        return "Hi. Tell me what you are craving or any allergies, and I can check the menu."
    return "I can only help with cafe menu questions and dietary safety for this menu."


def _format_restriction_acknowledgement(restrictions: CustomerRestrictions) -> str:
    """Build a deterministic acknowledgement for restriction-only turns.

    This response lets the session memory persist newly stated allergies or
    dietary modes without sending unrelated menu candidates to the model. It is
    used only after the router classifies the turn as dietary-safety context,
    not as a menu recommendation or section-browsing request.

    Args:
        restrictions (CustomerRestrictions):
            Current-turn restrictions extracted from the customer's latest
            message. Stored profile/session facts are not repeated here unless
            the customer mentioned them again in this turn.

    Returns:
        str:
            Customer-facing acknowledgement that confirms deterministic menu
            filtering will use the stated restrictions.
    """
    details: list[str] = []
    if restrictions.avoid_allergens:
        allergen_names = ", ".join(
            sorted(code.value.replace("_", " ").lower() for code in restrictions.avoid_allergens)
        )
        details.append(f"avoid {allergen_names}")
    if restrictions.modes:
        mode_names = ", ".join(
            sorted(mode.value.replace("_", " ").lower() for mode in restrictions.modes)
        )
        details.append(f"match {mode_names}")
    if restrictions.prefer_low_sugar:
        details.append("prefer lower-sugar options")
    if not details:
        return (
            "Tell me any allergies or dietary needs, and I'll filter the menu "
            "before suggesting items."
        )
    return (
        "Got it - I'll use this for the current session: "
        f"{'; '.join(details)}. I'll filter the menu before suggesting items, "
        "and unknown allergen data will be treated as unsafe."
    )

def _chunk_text(text: str, chunk_size: int = 24) -> list[str]:
    """Split immediate fallback text into simple streaming chunks.

    Args:
        text (str):
            Response text that should be yielded through the streaming endpoint.
        chunk_size (int):
            Maximum number of characters per chunk.

    Returns:
        list[str]:
            Ordered chunks that reconstruct the original text when joined.
    """
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


@dataclass(frozen=True, slots=True)
class _MenuSectionGroup:
    """Grouped safe menu entries for one customer-facing menu section.

    Args:
        name (str):
            Display name shown to the customer, such as `Coffees` or `Pizzas`.
        root_name (str):
            High-level source grouping, usually `Beverages`, `Food`, or `Menu`.
        entries (tuple[MenuBrowseItem, ...]):
            Safety-filtered browse entries that belong to the section.

    Returns:
        None:
            Dataclass instances are internal value objects used for formatting.
    """

    name: str
    root_name: str
    entries: tuple[MenuBrowseItem, ...]


def _format_menu_sections_response(
    entries: list[MenuBrowseItem],
    restrictions: CustomerRestrictions,
) -> str:
    """Build a restriction-aware menu overview grouped by section.

    Args:
        entries (list[MenuBrowseItem]):
            Safety-filtered browse entries used to derive available sections.
        restrictions (CustomerRestrictions):
            Active customer restrictions used to filter the menu before any
            section names or item counts are shown.

    Returns:
        str:
            Customer-facing section overview with item counts and a prompt to
            choose a section for the next turn. When restrictions are active,
            the copy explicitly says the list is filtered.
    """
    groups = _group_menu_sections(entries)
    if not groups:
        return (
            "I can't confirm a safe menu option from the available menu data. "
            "Please check with cafe staff before ordering."
        )

    restriction_note = _restriction_filter_note(restrictions)
    lines = [
        "Absolutely - here are the menu sections I can walk you through:",
    ]
    if restriction_note:
        lines.append(restriction_note)
    lines.append("")
    current_root = ""
    for group in groups:
        if group.root_name != current_root:
            if current_root:
                lines.append("")
            lines.append(f"{group.root_name}:")
            current_root = group.root_name
        item_word = "item" if len(group.entries) == 1 else "items"
        lines.append(f"- {group.name} ({len(group.entries)} {item_word})")
    lines.extend(
        [
            "",
            "Tell me a section, like 'coffees' or 'pizzas', and I'll show those items.",
        ]
    )
    return "\n".join(lines)


def _format_menu_section_items_response(
    group: _MenuSectionGroup,
    restrictions: CustomerRestrictions,
) -> str:
    """Build a restriction-aware item list for one selected menu section.

    Args:
        group (_MenuSectionGroup):
            Matched section containing only safety-filtered menu entries.
        restrictions (CustomerRestrictions):
            Active customer restrictions used to filter item names before they
            are shown in the selected section.

    Returns:
        str:
            Customer-facing item list for the selected section. When
            restrictions are active, the copy explicitly says unsafe or unknown
            items have been filtered out.
    """
    item_lines = [f"- {entry.item.name}" for entry in group.entries]
    restriction_note = _restriction_filter_note(restrictions)
    lines = [f"Of course - here's the {group.name} section:"]
    if restriction_note:
        lines.append(restriction_note)
    lines.extend(
        [
            "",
            *item_lines,
            "",
            "Want me to narrow that down by hot, iced, low sugar, or dietary needs?",
        ]
    )
    return "\n".join(lines)


def _restriction_filter_note(restrictions: CustomerRestrictions) -> str:
    """Describe active hard restrictions applied to a menu browse response.

    Args:
        restrictions (CustomerRestrictions):
            Customer restrictions currently applied by the deterministic safety
            filter before menu sections or item names are formatted.

    Returns:
        str:
            Empty string when no hard restrictions are active; otherwise a short
            customer-facing sentence that explains the displayed menu is already
            filtered and unknown allergen data is treated as unsafe.
    """
    details: list[str] = []
    if restrictions.avoid_allergens:
        allergen_names = ", ".join(
            sorted(code.value.replace("_", " ").lower() for code in restrictions.avoid_allergens)
        )
        details.append(f"avoiding {allergen_names}")
    if restrictions.modes:
        mode_names = ", ".join(
            sorted(mode.value.replace("_", " ").lower() for mode in restrictions.modes)
        )
        details.append(f"matching {mode_names}")
    if not details:
        return ""
    return (
        "I filtered this list for "
        f"{'; '.join(details)}. Items with unknown allergen data are not shown."
    )

def _group_menu_sections(entries: list[MenuBrowseItem]) -> list[_MenuSectionGroup]:
    """Group safe browse entries into customer-facing menu sections.

    Args:
        entries (list[MenuBrowseItem]):
            Safety-filtered entries with source category metadata.

    Returns:
        list[_MenuSectionGroup]:
            Ordered section groups preserving catalog order.
    """
    grouped: dict[str, tuple[str, str, list[MenuBrowseItem]]] = {}
    for entry in entries:
        root_name, section_name = _section_names_for_entry(entry)
        key = _section_key(section_name)
        if key not in grouped:
            grouped[key] = (section_name, root_name, [])
        grouped[key][2].append(entry)
    return [
        _MenuSectionGroup(name=name, root_name=root_name, entries=tuple(group_entries))
        for name, root_name, group_entries in grouped.values()
    ]


def _match_menu_section(
    entries: list[MenuBrowseItem],
    message: str,
) -> _MenuSectionGroup | None:
    """Match customer text to one available menu section.

    Args:
        entries (list[MenuBrowseItem]):
            Safety-filtered browse entries used to derive matchable sections.
        message (str):
            Raw customer message from the active turn.

    Returns:
        _MenuSectionGroup | None:
            Best matching section when the text clearly names one; otherwise None.
    """
    query_tokens = _section_tokens(message) - _SECTION_FILLER_TOKENS
    if not query_tokens:
        return None

    best_group: _MenuSectionGroup | None = None
    best_score = 0
    for group in _group_menu_sections(entries):
        aliases = [_section_tokens(group.name)]
        aliases.extend(_section_tokens(entry.category_name) for entry in group.entries)
        aliases.extend(_section_tokens(entry.category_path) for entry in group.entries)
        for alias_tokens in aliases:
            if not alias_tokens:
                continue
            overlap = len(query_tokens & alias_tokens)
            if overlap == 0:
                continue
            exact = query_tokens == alias_tokens
            subset = query_tokens.issubset(alias_tokens) or alias_tokens.issubset(query_tokens)
            if not exact and not subset:
                continue
            score = (100 if exact else 0) + (50 if subset else 0) + overlap * 10 + len(alias_tokens)
            if score > best_score:
                best_score = score
                best_group = group
    return best_group


def _section_names_for_entry(entry: MenuBrowseItem) -> tuple[str, str]:
    """Choose display root and section names from a source category path.

    Args:
        entry (MenuBrowseItem):
            Browse entry with category metadata from catalog or legacy tables.

    Returns:
        tuple[str, str]:
            `(root_name, section_name)` suitable for customer-facing grouping.
    """
    parts = [part.strip() for part in entry.category_path.split(">") if part.strip()]
    if not parts:
        return "Menu", entry.category_name or "Other"
    root_name = parts[0] if len(parts) > 1 else "Menu"
    meaningful_parts = parts[1:] if len(parts) > 1 else parts
    section_name = meaningful_parts[0] if meaningful_parts else entry.category_name or "Other"
    return root_name, section_name


def _section_key(section_name: str) -> str:
    """Normalize a display section name into a stable grouping key.

    Args:
        section_name (str):
            Customer-facing section name.

    Returns:
        str:
            Stable lowercase key built from singularized tokens.
    """
    return " ".join(sorted(_section_tokens(section_name)))


def _section_tokens(text: str) -> set[str]:
    """Tokenize and lightly singularize section matching text.

    Args:
        text (str):
            Raw section, category, path, or customer text.

    Returns:
        set[str]:
            Normalized tokens used for deterministic section matching.
    """
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) > 4 and token.endswith("ies"):
            token = f"{token[:-3]}y"
        elif len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes", "zes")):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _might_be_menu_section_request(message: str) -> bool:
    """Return whether a message is likely asking to open a menu section.

    Args:
        message (str):
            Raw customer message from the active turn.

    Returns:
        bool:
            True for explicit section/category wording, natural browse
            questions such as `what sandwiches do you have`, restriction-aware
            browse requests such as `I am allergic to dairy, show me coffees`,
            or short section-like replies such as `coffees` or `pizzas`.
    """
    tokens = _section_tokens(message)
    if not tokens:
        return False
    if tokens & {"section", "category", "categorie"}:
        return True
    smalltalk_tokens = {"hi", "hello", "hey", "thank", "thanks", "bye"}
    if tokens & smalltalk_tokens:
        return False
    recommendation_tokens = {"get", "order", "recommend", "should", "suggest", "try"}
    if tokens & recommendation_tokens:
        return False
    browse_tokens = {
        "available",
        "do",
        "have",
        "list",
        "see",
        "show",
        "what",
        "which",
    }
    if tokens & browse_tokens:
        return True
    safety_tokens = {
        "allergic",
        "allergy",
        "avoid",
        "avoiding",
        "dairy",
        "egg",
        "gluten",
        "peanut",
        "soy",
        "vegan",
        "vegetarian",
    }
    if tokens & safety_tokens:
        return False
    return 0 < len(tokens - _SECTION_FILLER_TOKENS) <= 2


_SECTION_FILLER_TOKENS = {
    "a",
    "about",
    "allergic",
    "allergy",
    "am",
    "and",
    "are",
    "available",
    "avoid",
    "avoiding",
    "can",
    "could",
    "do",
    "does",
    "for",
    "give",
    "has",
    "have",
    "i",
    "in",
    "is",
    "item",
    "list",
    "me",
    "of",
    "option",
    "please",
    "see",
    "show",
    "the",
    "there",
    "to",
    "today",
    "want",
    "what",
    "which",
    "with",
    "yes",
    "you",
}


def _is_broad_menu_request(message: str) -> bool:
    """Return whether a message asks for a general menu browse.

    Args:
        message (str):
            Raw customer text from the current turn.

    Returns:
        bool:
            True when the message asks to browse, show, list, or recommend the
            menu in general, allowing the agent to return safety-filtered
            catalog items instead of an empty retrieval result.
    """
    tokens = {token.strip("?.!,;:").lower() for token in message.split()}
    tokens.discard("")
    menu_terms = {"menu", "menus", "options", "items"}
    browse_terms = {
        "browse",
        "give",
        "have",
        "list",
        "recommend",
        "see",
        "show",
        "suggest",
        "view",
    }
    filler_terms = {
        "a",
        "all",
        "an",
        "available",
        "can",
        "could",
        "entire",
        "full",
        "me",
        "please",
        "some",
        "the",
        "then",
        "today",
        "us",
        "you",
    }
    allowed_terms = menu_terms | browse_terms | filler_terms
    return bool(tokens & menu_terms) and bool(tokens & browse_terms) and tokens.issubset(
        allowed_terms
    )

def _query_with_preferences(message: str, preferences: dict[str, object]) -> str:
    """Augment a retrieval query with safe durable preference hints.

    Args:
        message (str):
            Current user message used as the base retrieval query.
        preferences (dict[str, object]):
            Durable preference facts loaded for the recognized customer.

    Returns:
        str:
            Query text with a milk preference appended when one is available.
    """
    milk_preference = preferences.get("milk_preference")
    if isinstance(milk_preference, str) and milk_preference:
        return f"{message} {milk_preference}"
    return message
