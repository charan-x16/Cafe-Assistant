"""Message classification router for the cafe assistant state machine.

Routing is intentionally simple and conservative. Deterministic rules handle
clear menu, safety, smalltalk, and out-of-scope messages first. Model-based
classification is used only when rules are low-confidence, and untrusted user
text is wrapped before it reaches a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from cafe_assistant.gateway.model_gateway import ChatMessage, ChatProvider
from cafe_assistant.observability.tracing import span, token_count
from cafe_assistant.security.injection import wrap_untrusted_text


class Intent(StrEnum):
    """Supported high-level routes for one user message."""

    MENU_QA = "menu_qa"
    RECOMMENDATION = "recommendation"
    DIETARY_SAFETY = "dietary_safety"
    SMALLTALK = "smalltalk"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Router decision and confidence score."""

    intent: Intent
    confidence: float


_MENU_PATTERN = re.compile(
    r"\b(menu|drink|coffee|tea|latte|mocha|espresso|cappuccino|sandwich|pastry|cookie|"
    r"muffin|croissant|toast|danish|chai|matcha|cold brew|recommend|suggest|have|get|order)\b",
    re.IGNORECASE,
)
_SAFETY_PATTERN = re.compile(
    r"\b(allerg|peanut|tree nut|dairy|gluten|soy|egg|vegan|vegetarian|gluten[- ]?free)\b",
    re.IGNORECASE,
)
_SMALLTALK_PATTERN = re.compile(r"\b(hi|hello|hey|thanks|thank you)\b", re.IGNORECASE)
_OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(weather|sports|stock|movie|flight|hotel|news|politics)\b",
    re.IGNORECASE,
)


class MessageRouter:
    """Classify messages before retrieval and composition."""

    def __init__(self, cheap_model: ChatProvider, strong_model: ChatProvider) -> None:
        """Create a router with cheap and strong model fallbacks.

        Args:
            cheap_model (ChatProvider):
                Low-cost provider used first when rules are low-confidence.
            strong_model (ChatProvider):
                Stronger provider used only when cheap classification is unclear.

        Returns:
            None:
                The router stores the providers for future classification calls.
        """
        self.cheap_model = cheap_model
        self.strong_model = strong_model

    async def classify(self, message: str) -> ClassificationResult:
        """Classify a user message into one agent route.

        Args:
            message (str):
                Current user message. It is treated as untrusted when sent to a
                model classifier.

        Returns:
            ClassificationResult:
                Intent and confidence selected by rules, cheap model, or strong
                model in that order.
        """
        with span("router.classify", route="unknown", confidence=0.0) as record:
            rules_result = self._classify_with_rules(message)
            result = rules_result
            source = "rules"
            if rules_result.confidence < 0.6:
                cheap_result = await self._classify_with_model(self.cheap_model, message)
                if cheap_result is not None and cheap_result.confidence >= 0.6:
                    result = cheap_result
                    source = "cheap_model"
                else:
                    strong_result = await self._classify_with_model(self.strong_model, message)
                    if strong_result is not None:
                        result = strong_result
                        source = "strong_model"
            record.attributes.update(
                {
                    "route": result.intent.value,
                    "confidence": result.confidence,
                    "source": source,
                }
            )
            return result

    def _classify_with_rules(self, message: str) -> ClassificationResult:
        """Classify clear messages with deterministic keyword rules.

        Args:
            message (str):
                Current user message.

        Returns:
            ClassificationResult:
                Rule-based route. Low-confidence default goes to menu QA so the
                model cascade can disambiguate before orchestration continues.
        """
        if _OUT_OF_SCOPE_PATTERN.search(message):
            return ClassificationResult(Intent.OUT_OF_SCOPE, 0.85)
        if _SAFETY_PATTERN.search(message):
            if _MENU_PATTERN.search(message):
                return ClassificationResult(Intent.RECOMMENDATION, 0.75)
            return ClassificationResult(Intent.DIETARY_SAFETY, 0.75)
        if _MENU_PATTERN.search(message):
            if "recommend" in message.lower() or "suggest" in message.lower():
                return ClassificationResult(Intent.RECOMMENDATION, 0.8)
            return ClassificationResult(Intent.MENU_QA, 0.72)
        if _SMALLTALK_PATTERN.search(message):
            return ClassificationResult(Intent.SMALLTALK, 0.7)
        return ClassificationResult(Intent.MENU_QA, 0.45)

    async def _classify_with_model(
        self,
        provider: ChatProvider,
        message: str,
    ) -> ClassificationResult | None:
        """Ask one chat provider to classify a low-confidence message.

        Args:
            provider (ChatProvider):
                Chat provider used for this classifier attempt.
            message (str):
                Current user message, wrapped as untrusted data in the prompt.

        Returns:
            ClassificationResult | None:
                Parsed route with conservative confidence when the provider emits
                a supported intent string, otherwise None so the caller can fall
                back to the next classifier or rule result.
        """
        prompt = (
            "Classify this cafe-assistant message as one of "
            "menu_qa, recommendation, dietary_safety, smalltalk, out_of_scope. "
            f"Message: {wrap_untrusted_text('user_message', message)}"
        )
        chunks: list[str] = []
        with span(
            "llm.classify",
            model="cheap_or_strong",
            prompt_version="classifier_v1",
            input_tokens=token_count(prompt),
        ) as record:
            async for chunk in provider.stream_chat(
                [ChatMessage(role="user", content=prompt)],
                timeout_seconds=2.0,
            ):
                chunks.append(chunk)
            record.attributes["output_tokens"] = token_count("".join(chunks))
        text = "".join(chunks).strip().lower()
        for intent in Intent:
            if intent.value in text:
                return ClassificationResult(intent, 0.65)
        return None