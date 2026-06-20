"""Implementation module for versioning.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from cafe_assistant.config import settings


@dataclass(frozen=True, slots=True)
class ComponentVersion:
    """Container for component version behavior and data."""
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class VersionRegistry:
    """Container for version registry behavior and data."""
    prompts: dict[str, str]
    tools: dict[str, str]
    retrievers: dict[str, str]
    embedding_model: ComponentVersion
    model_choices: dict[str, str]
    policy_rules: dict[str, str]
    memory_write_rules: dict[str, str]
    orchestrator_graph: ComponentVersion

    def as_trace_attributes(self) -> dict[str, object]:
        """Handle as trace attributes.

        Args:
            None.

        Returns:
            dict[str, object]:
                Value produced for the caller according to the function contract.
        """
        return {
            "prompts": dict(self.prompts),
            "tools": dict(self.tools),
            "retrievers": dict(self.retrievers),
            "embedding_model": {
                "name": self.embedding_model.name,
                "version": self.embedding_model.version,
            },
            "model_choices": dict(self.model_choices),
            "policy_rules": dict(self.policy_rules),
            "memory_write_rules": dict(self.memory_write_rules),
            "orchestrator_graph": {
                "name": self.orchestrator_graph.name,
                "version": self.orchestrator_graph.version,
            },
        }


def get_version_registry() -> VersionRegistry:
    """Return version registry.

    Args:
        None.

    Returns:
        VersionRegistry:
            Value produced for the caller according to the function contract.
    """
    return VersionRegistry(
        prompts={
            "classifier": "classifier_v1",
            "composer": "composer_v1",
        },
        tools={
            "menu_lookup": "menu_lookup_v1",
            "search_menu": "search_menu_v1",
            "dietary_filter": "dietary_filter_v1",
        },
        retrievers={
            "keyword": "postgres_fts_trigram_v1",
            "semantic": f"{settings.vector_provider}_cosine_v1",
            "vector_collection": settings.qdrant_collection,
            "hybrid": "rrf_v1",
        },
        embedding_model=ComponentVersion(
            name=settings.embedding_model_name,
            version=(
                f"{settings.embedding_provider}:{settings.embedding_model_name}"
                f"_dim_{settings.embedding_dimension}_v1"
            ),
        ),
        model_choices={
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "cheap": settings.cheap_chat_provider,
            "strong": settings.strong_chat_provider,
            "default_chat_model": settings.default_chat_model_name,
        },
        policy_rules={
            "dietary_safety": "unknown_unsafe_v1",
            "medical_refusal": "medical_refusal_v1",
            "prompt_injection": "neutralize_delimit_v1",
            "tenant_isolation": "tenant_scoped_access_v1",
        },
        memory_write_rules={
            "preference_auto_write": "preference_auto_write_v1",
            "health_data_consent_gate": "dietary_health_consent_v1",
        },
        orchestrator_graph=ComponentVersion(
            name="custom_fsm",
            version="classified_retrieving_filtering_recommending_composing_v1",
        ),
    )
