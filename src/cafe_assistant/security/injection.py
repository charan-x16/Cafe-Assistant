"""Prompt-injection neutralization for untrusted user, menu, and policy text.

The assistant treats customer messages and catalog/policy content as data, never
as instructions. These helpers neutralize common instruction-takeover phrases,
wrap untrusted text in explicit delimiters, and verify that dangerous instruction
patterns do not reach non-system model messages. The guard is intentionally
conservative: suspicious text is replaced instead of interpreted.
"""

from __future__ import annotations

import re

_INSTRUCTION_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior|earlier|above)"
        r"(?:\s+instructions?)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard\s+(?:all\s+)?(?:prior|previous|earlier|above)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bforget\s+(?:all\s+)?(?:prior|previous|earlier)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(
        r"\bnew\s+(?:system|developer)\s+(?:prompt|message|instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:system|developer)\s*(?:prompt|message|instructions?)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:system|developer|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"</?\s*(?:system|developer|assistant)\s*>", re.IGNORECASE),
    re.compile(r"\breveal\s+(?:the\s+)?(?:prompt|instructions?|secrets?)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:print|show|dump|exfiltrate)\s+(?:the\s+)?"
        r"(?:system|developer)?\s*(?:prompt|instructions?|secrets?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfollow\s+(?:these|the\s+following|my)\s+instructions?\b", re.IGNORECASE),
    re.compile(
        r"\boverride\s+(?:the\s+)?(?:system|developer|safety|policy|rules?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdo\s+not\s+(?:obey|follow)\s+(?:the\s+)?"
        r"(?:system|developer|previous|safety)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bprompt\s+injection\b", re.IGNORECASE),
)

UNTRUSTED_START = "<UNTRUSTED_DATA>"
UNTRUSTED_END = "</UNTRUSTED_DATA>"
_NEUTRALIZED = "[neutralized instruction text]"


def neutralize_instruction_patterns(text: str) -> str:
    """Replace instruction-like phrases inside untrusted text.

    Args:
        text (str):
            Untrusted user, menu, policy, or preference text that may contain
            instructions aimed at the model instead of the assistant user.

    Returns:
        str:
            Text with known instruction-takeover patterns replaced by a neutral marker.
    """
    neutralized = text
    for pattern in _INSTRUCTION_PATTERNS:
        neutralized = pattern.sub(_NEUTRALIZED, neutralized)
    return neutralized


def wrap_untrusted_text(label: str, text: str) -> str:
    """Wrap untrusted text in explicit delimiters after neutralization.

    Args:
        label (str):
            Short source label such as `user_message` or `menu_item_name`.
        text (str):
            Raw untrusted text to send as model-visible data.

    Returns:
        str:
            Delimited text block with a normalized label and neutralized content.
    """
    safe_label = re.sub(r"[^A-Z0-9_]", "_", label.upper()).strip("_") or "TEXT"
    neutralized = neutralize_instruction_patterns(text)
    return f"{UNTRUSTED_START} {safe_label}\n{neutralized}\n{UNTRUSTED_END}"


def assert_model_context_guarded(messages: list[object]) -> None:
    """Reject model messages where untrusted instruction text escaped neutralization.

    Args:
        messages (list[object]):
            Chat-message-like objects with `role` and `content` attributes.

    Returns:
        None:
            The function returns only when non-system message content is guarded.
    """
    for message in messages:
        role = str(getattr(message, "role", "")).lower()
        if role == "system":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        if any(pattern.search(content) for pattern in _INSTRUCTION_PATTERNS):
            raise PromptInjectionGuardError("Untrusted instruction pattern reached model context.")


class PromptInjectionGuardError(ValueError):
    """Raised when untrusted instruction text reaches a non-system model message."""
