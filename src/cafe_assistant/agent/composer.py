"""Response composition boundary for safe menu answers.

The composer receives only the final `safe_items` selected by deterministic
retrieval and filtering. It builds guarded chat messages, wraps all untrusted
user/menu/preference text, records model cost metrics, and stores the last prompt
messages for tests that verify unsafe items never reach model context.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from cafe_assistant.config import settings
from cafe_assistant.domain.dietary import CustomerRestrictions, MenuItemView
from cafe_assistant.gateway.model_gateway import ChatMessage, ChatProvider
from cafe_assistant.observability.metrics import record_llm_cost
from cafe_assistant.observability.tracing import estimate_cost, span, token_count
from cafe_assistant.security.injection import (
    assert_model_context_guarded,
    neutralize_instruction_patterns,
    wrap_untrusted_text,
)

PROMPT_VERSION = "composer_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.txt"


@dataclass(frozen=True, slots=True)
class ComposeInput:
    """Grounded input allowed into response composition."""

    user_message: str
    safe_items: list[MenuItemView]
    restrictions: CustomerRestrictions
    preferences: dict[str, object] | None = None
    include_medical_disclaimer: bool = False


class ResponseComposer:
    """Build and stream final natural-language responses from safe item context."""

    def __init__(self, strong_model: ChatProvider) -> None:
        """Create a composer backed by the strong chat provider.

        Args:
            strong_model (ChatProvider):
                Provider used for final answer synthesis.

        Returns:
            None:
                The composer is ready to build guarded prompts.
        """
        self.strong_model = strong_model
        self.last_messages: list[ChatMessage] = []

    async def stream(
        self,
        compose_input: ComposeInput,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream a composed answer from the strong model.

        Args:
            compose_input (ComposeInput):
                User message, active restrictions, preferences, and safe items.
            timeout_seconds (float):
                Remaining request time allowed for model streaming.

        Returns:
            AsyncIterator[str]:
                Model tokens as they arrive. Metrics and trace attributes are
                recorded after streaming completes.
        """
        messages = self._build_messages(compose_input)
        self.last_messages = messages
        input_text = "\n".join(message.content for message in messages)
        input_tokens = token_count(input_text)
        output_chunks: list[str] = []
        with span(
            "llm.compose",
            model=settings.default_chat_model_name,
            prompt_version=PROMPT_VERSION,
            input_tokens=input_tokens,
            safe_item_ids=[item.id for item in compose_input.safe_items],
            safe_item_names=[item.name for item in compose_input.safe_items],
            prompt_messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        ) as record:
            async for token in self.strong_model.stream_chat(
                messages,
                timeout_seconds=timeout_seconds,
            ):
                output_chunks.append(token)
                yield token
            output_text = "".join(output_chunks)
            output_tokens = token_count(output_text)
            cost = estimate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_per_1k=settings.strong_model_input_cost_per_1k,
                output_cost_per_1k=settings.strong_model_output_cost_per_1k,
            )
            record.attributes.update(
                {
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                }
            )
            record_llm_cost(
                cost,
                model=settings.default_chat_model_name,
                prompt_version=PROMPT_VERSION,
            )

    async def compose(self, compose_input: ComposeInput, *, timeout_seconds: float) -> str:
        """Return the full composed answer as a string.

        Args:
            compose_input (ComposeInput):
                Grounded composition input containing only safe menu items.
            timeout_seconds (float):
                Remaining request time allowed for model streaming.

        Returns:
            str:
                Concatenated model output.
        """
        chunks: list[str] = []
        async for token in self.stream(compose_input, timeout_seconds=timeout_seconds):
            chunks.append(token)
        return "".join(chunks)

    def _build_messages(self, compose_input: ComposeInput) -> list[ChatMessage]:
        """Build guarded chat messages for final answer synthesis.

        Args:
            compose_input (ComposeInput):
                Grounded composition input containing only safe menu items.

        Returns:
            list[ChatMessage]:
                System prompt and user-context message. All untrusted values are
                wrapped or neutralized before returning.
        """
        safe_lines = [
            f"SAFE_ITEM: {neutralize_instruction_patterns(item.name)} | "
            f"sugar_grams={item.sugar_grams} | "
            f"wrapped_name={wrap_untrusted_text('menu_item_name', item.name)}"
            for item in compose_input.safe_items
        ]
        avoid_allergens = sorted(
            allergen.value for allergen in compose_input.restrictions.avoid_allergens
        )
        restrictions_line = (
            "Restrictions: "
            f"avoid_allergens={avoid_allergens}, "
            f"modes={sorted(mode.value for mode in compose_input.restrictions.modes)}, "
            f"prefer_low_sugar={compose_input.restrictions.prefer_low_sugar}"
        )
        preferences = compose_input.preferences or {}
        preferences_line = (
            f"Preferences: {wrap_untrusted_text('preferences', str(preferences))}"
            if preferences
            else "Preferences: none"
        )
        disclaimer = (
            "Medical disclaimer: This is not medical advice; please check with a clinician."
            if compose_input.include_medical_disclaimer
            else "Medical disclaimer: none"
        )
        user_context = "\n".join(
            [
                f"User message: {wrap_untrusted_text('user_message', compose_input.user_message)}",
                restrictions_line,
                preferences_line,
                disclaimer,
                *safe_lines,
            ]
        )
        messages = [
            ChatMessage(role="system", content=_PROMPT_PATH.read_text(encoding="utf-8")),
            ChatMessage(role="user", content=user_context),
        ]
        assert_model_context_guarded(messages)
        return messages