from __future__ import annotations

import re

_INSTRUCTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?previous(?:\s+instructions)?\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bsystem\s*prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s*message\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(?:the\s+)?(?:prompt|instructions|secrets?)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:prior|previous)\b", re.IGNORECASE),
)

UNTRUSTED_START = "<UNTRUSTED_DATA>"
UNTRUSTED_END = "</UNTRUSTED_DATA>"


def neutralize_instruction_patterns(text: str) -> str:
    neutralized = text
    for pattern in _INSTRUCTION_PATTERNS:
        neutralized = pattern.sub("[neutralized instruction text]", neutralized)
    return neutralized


def wrap_untrusted_text(label: str, text: str) -> str:
    safe_label = re.sub(r"[^A-Z0-9_]", "_", label.upper()).strip("_") or "TEXT"
    neutralized = neutralize_instruction_patterns(text)
    return f"{UNTRUSTED_START} {safe_label}\n{neutralized}\n{UNTRUSTED_END}"


def assert_model_context_guarded(messages: list[object]) -> None:
    for message in messages:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        lowered = content.lower()
        if any(pattern.search(lowered) for pattern in _INSTRUCTION_PATTERNS):
            raise PromptInjectionGuardError("Untrusted instruction pattern reached model context.")


class PromptInjectionGuardError(ValueError):
    pass
