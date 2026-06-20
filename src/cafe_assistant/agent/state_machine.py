"""Explicit chat-agent state machine for safe menu recommendations.

The state machine routes each request, carries session/profile restrictions,
invokes deterministic menu tools, and passes only safety-filtered menu items to
response composition. It enforces the rule that LLM synthesis never sees raw
menu candidates or decides allergen/dietary safety.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum

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
from cafe_assistant.db.repositories.profile_repo import get_customer
from cafe_assistant.domain.dietary import CustomerRestrictions, MenuItemView
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
from cafe_assistant.memory.write_gate import classify_candidate_writes, persist_allowed_writes
from cafe_assistant.observability.metrics import record_quality_event
from cafe_assistant.observability.tracing import span, start_trace
from cafe_assistant.security.audit import AuditContext, append_audit_event


class AgentState(StrEnum):
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
    session_id: str
    tenant_id: int
    message: str
    device_token: str | None = None
    customer_id: int | None = None
    request_id: str = "internal"
    trace_id: str = "internal"
    actor: str = "anonymous"


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_tool_calls: int = settings.agent_max_tool_calls
    deadline_seconds: float = settings.agent_deadline_seconds
    search_k: int = 8


@dataclass(slots=True)
class ChatAgentResult:
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
    response: str | None
    safe_items: list[MenuItemView]
    restrictions: CustomerRestrictions
    preferences: dict[str, object]
    state_history: list[AgentState]
    medical_disclaimer: bool
    tool_calls: int
    customer_id: int | None


class ToolBudgetExceededError(RuntimeError):
    pass


class RequestDeadlineExceededError(RuntimeError):
    pass


class RecommenderUnavailableError(RuntimeError):
    pass


class ChatAgent:
    def __init__(
        self,
        session: AsyncSession,
        *,
        memory: SessionMemory | None = None,
        chat_models: ChatModelCascade | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.session = session
        self.memory = memory or get_redis_session_memory()
        self.chat_models = chat_models or get_chat_model_cascade()
        self.embedding_provider = embedding_provider
        self.config = config or AgentConfig()
        self.router = MessageRouter(self.chat_models.cheap, self.chat_models.strong)
        self.tools = ToolRegistry(session, embedding_provider=embedding_provider)
        self.composer = ResponseComposer(self.chat_models.strong)

    async def run(self, request: ChatAgentRequest) -> ChatAgentResult:
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
        except Exception as exc:  # noqa: BLE001 - user-facing fallback must catch all failures.
            record_quality_event("errors_agent_total")
            return ChatAgentResult(
                response=f"Sorry, I could not complete that safely right now. ({exc})",
                state_history=[AgentState.FAILED],
            )

    async def stream_response(self, request: ChatAgentRequest) -> AsyncIterator[str]:
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
            )
            await self._audit_recommendation(request, prepared)
        except Exception:
            yield "Sorry, I could not complete that safely right now."

    async def _prepare_response(
        self,
        request: ChatAgentRequest,
        *,
        deadline_at: float,
    ) -> _PreparedResponse:
        state_history: list[AgentState] = []
        tool_calls = 0

        try:
            session_state = await self.memory.load(request.session_id)
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
        search_query = _query_with_preferences(request.message, memory_context.preferences)

        self._ensure_deadline(deadline_at)
        classification = await self.router.classify(request.message)
        state_history.append(AgentState.CLASSIFIED)

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
                preferences=memory_context.preferences,
                state_history=[*state_history, AgentState.ESCALATED],
                medical_disclaimer=True,
                tool_calls=tool_calls,
                customer_id=customer_id,
            )

        if classification.intent in {Intent.OUT_OF_SCOPE, Intent.SMALLTALK}:
            response = _non_menu_response(classification.intent)
            return _PreparedResponse(
                response=response,
                safe_items=[],
                restrictions=extraction.restrictions,
                preferences=memory_context.preferences,
                state_history=[*state_history, AgentState.COMPLETE],
                medical_disclaimer=False,
                tool_calls=tool_calls,
                customer_id=customer_id,
            )

        self._ensure_deadline(deadline_at)
        state_history.append(AgentState.RETRIEVING)
        tool_calls += 1
        self._ensure_tool_budget(tool_calls)
        lookup_output = await self._safe_menu_lookup(request, extraction.restrictions)
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
                preferences=memory_context.preferences,
                state_history=[*state_history, AgentState.FILTERING, AgentState.COMPLETE],
                medical_disclaimer=False,
                tool_calls=tool_calls,
                customer_id=customer_id,
            )

        tool_calls += 1
        self._ensure_tool_budget(tool_calls)
        output = await self._safe_search_menu(
            request,
            query=search_query,
            restrictions=extraction.restrictions,
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
            response = (
                "I can't confirm a safe menu option from the available menu data. "
                "Please check with cafe staff before ordering."
            )
            record_quality_event("empty_safe_sets_total", reason="filter_result_empty")
            return _PreparedResponse(
                response=response,
                safe_items=[],
                restrictions=extraction.restrictions,
                preferences=memory_context.preferences,
                state_history=[*state_history, AgentState.COMPLETE],
                medical_disclaimer=False,
                tool_calls=tool_calls,
                customer_id=customer_id,
            )

        state_history.append(AgentState.RECOMMENDING)
        recommended_items = safe_items[:3]
        state_history.append(AgentState.COMPOSING)
        return _PreparedResponse(
            response=None,
            safe_items=recommended_items,
            restrictions=extraction.restrictions,
            preferences=memory_context.preferences,
            state_history=state_history,
            medical_disclaimer=False,
            tool_calls=tool_calls,
            customer_id=customer_id,
        )

    async def _safe_menu_lookup(
        self,
        request: ChatAgentRequest,
        restrictions: CustomerRestrictions,
    ) -> MenuItemsOutput:
        """Run the exact lookup tool with active safety restrictions.

        Args:
            request (ChatAgentRequest):
                Chat request containing tenant, session, and user message data.
            restrictions (CustomerRestrictions):
                Customer restrictions extracted for the current turn and session.

        Returns:
            MenuItemsOutput:
                Safety-filtered lookup results, or an empty output if lookup fails.
        """
        try:
            output = await self.tools.call(
                "menu_lookup",
                MenuLookupInput(
                    tenant_id=request.tenant_id,
                    query=request.message,
                    restrictions=RestrictionsSchema.from_domain(restrictions),
                    limit=3,
                ),
            )
            if not isinstance(output, MenuItemsOutput):
                raise TypeError("menu_lookup tool returned an unexpected output type.")
            return output
        except Exception:
            record_quality_event("recommender_fallback_total", stage="menu_lookup")
            return MenuItemsOutput(items=[])

    async def _safe_search_menu(
        self,
        request: ChatAgentRequest,
        *,
        query: str,
        restrictions: CustomerRestrictions,
    ) -> MenuItemsOutput:
        try:
            output = await self.tools.call(
                "search_menu",
                SearchMenuInput(
                    tenant_id=request.tenant_id,
                    query=query,
                    restrictions=RestrictionsSchema.from_domain(restrictions),
                    k=self.config.search_k,
                ),
            )
            if not isinstance(output, MenuItemsOutput):
                raise TypeError("search_menu tool returned an unexpected output type.")
            return output
        except Exception:
            record_quality_event("recommender_fallback_total", stage="search_menu")
            return await self._fallback_popular_items(request, restrictions)

    async def _fallback_popular_items(
        self,
        request: ChatAgentRequest,
        restrictions: CustomerRestrictions,
    ) -> MenuItemsOutput:
        """Load popular fallback items and keep the deterministic safety gate active.

        Args:
            request (ChatAgentRequest):
                Chat request that supplies the tenant scope for fallback lookup.
            restrictions (CustomerRestrictions):
                Customer restrictions that must still be enforced during fallback.

        Returns:
            MenuItemsOutput:
                Safety-filtered fallback items capped to the configured search size.
        """
        all_items = await self.tools.menu_lookup(
            MenuLookupInput(
                tenant_id=request.tenant_id,
                query="",
                restrictions=RestrictionsSchema.from_domain(restrictions),
                limit=50,
            )
        )
        if not isinstance(all_items, MenuItemsOutput):
            raise RecommenderUnavailableError("Fallback menu lookup failed.")
        result = await self.tools.dietary_filter(
            DietaryFilterInput(
                items=[
                    MenuItemViewSchema.from_domain(item.to_domain())
                    for item in all_items.items
                ],
                restrictions=RestrictionsSchema.from_domain(restrictions),
            )
        )
        if not isinstance(result, MenuItemsOutput):
            raise RecommenderUnavailableError("Fallback dietary filter failed.")
        return MenuItemsOutput(items=result.items[: self.config.search_k])

    async def _resolve_customer_id(self, request: ChatAgentRequest) -> int | None:
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
    ) -> None:
        try:
            current_state = await self.memory.load(request.session_id)
        except Exception:
            current_state = SessionState()
        state = SessionState(
            restrictions=restrictions,
            recent_turns=current_state.recent_turns,
        )
        try:
            await self.memory.save(
                request.session_id,
                append_turns(state, request.message, response),
            )
        except Exception:
            record_quality_event("memory_unavailable_total")
        if customer_id is None:
            return

        writes = classify_candidate_writes(
            message=request.message,
            restrictions=restrictions,
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
                    },
                )

    async def _audit_recommendation(
        self,
        request: ChatAgentRequest,
        prepared: _PreparedResponse,
    ) -> None:
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
            },
        )

    def _audit_context(
        self,
        request: ChatAgentRequest,
        *,
        customer_id: int | None,
    ) -> AuditContext:
        return AuditContext(
            tenant_id=request.tenant_id,
            actor=f"customer:{customer_id}" if customer_id is not None else request.actor,
            request_id=request.request_id,
            trace_id=request.trace_id,
        )

    def _ensure_tool_budget(self, tool_calls: int) -> None:
        if tool_calls > self.config.max_tool_calls:
            raise ToolBudgetExceededError("Agent tool-call budget exceeded.")

    def _ensure_deadline(self, deadline_at: float) -> None:
        if asyncio.get_running_loop().time() > deadline_at:
            raise RequestDeadlineExceededError("Agent request deadline exceeded.")

    def _remaining_seconds(self, deadline_at: float) -> float:
        remaining = deadline_at - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RequestDeadlineExceededError("Agent request deadline exceeded.")
        return remaining


def _non_menu_response(intent: Intent) -> str:
    if intent == Intent.SMALLTALK:
        return "Hi. Tell me what you are craving or any allergies, and I can check the menu."
    return "I can only help with cafe menu questions and dietary safety for this menu."


def _chunk_text(text: str, chunk_size: int = 24) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _query_with_preferences(message: str, preferences: dict[str, object]) -> str:
    milk_preference = preferences.get("milk_preference")
    if isinstance(milk_preference, str) and milk_preference:
        return f"{message} {milk_preference}"
    return message
