from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from cafe_assistant.api.routes_chat import router as chat_router
from cafe_assistant.api.routes_consent import router as consent_router
from cafe_assistant.api.routes_observability import router as observability_router
from cafe_assistant.config import settings
from cafe_assistant.gateway.model_gateway import get_embedding_provider
from cafe_assistant.security.redaction import configure_redacted_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown hooks for the FastAPI application.

    Args:
        app (FastAPI):
            Application instance managed by the ASGI server.

    Returns:
        AsyncIterator[None]:
            Lifespan context yielded to FastAPI after startup warmup finishes.
    """
    del app
    await _warm_embedding_provider()
    yield


async def _warm_embedding_provider() -> None:
    """Load and exercise the configured embedding provider during startup.

    Args:
        None:
            The provider is resolved from environment-backed settings.

    Returns:
        None:
            Warmup failures are logged and never prevent the API from starting.
    """
    try:
        await asyncio.to_thread(_load_embedding_provider)
        logger.info("Embedding provider warmed: %s", settings.embedding_model_name)
    except Exception as exc:  # noqa: BLE001 - warmup must not block health startup.
        logger.warning("Embedding provider warmup failed: %s", type(exc).__name__)


def _load_embedding_provider() -> None:
    """Build the cached embedding provider and run one small warmup embed.

    Args:
        None:
            The provider cache is owned by `get_embedding_provider`.

    Returns:
        None:
            The configured model is loaded into process memory for later retrieval calls.
    """
    provider = get_embedding_provider()
    provider.embed(["startup embedding warmup"])


def create_app() -> FastAPI:
    """Create the configured FastAPI application.

    Args:
        None:
            Application settings are read from environment-backed configuration.

    Returns:
        FastAPI:
            API application with chat, identity, observability, health, and startup warmup.
    """
    configure_redacted_logging()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(consent_router)
    app.include_router(observability_router)

    @app.get("/")
    async def root() -> RedirectResponse:
        """Redirect browser visitors from the API root to the chat UI.

        Args:
            None:
                The route does not accept input parameters.

        Returns:
            RedirectResponse:
                Temporary redirect to the static chat page served at `/chat`.
        """
        return RedirectResponse(url="/chat", status_code=307)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return a minimal process health response.

        Args:
            None:
                The route does not accept input parameters.

        Returns:
            dict[str, str]:
                Static health payload used by local checks and load balancers.
        """
        return {"status": "ok"}

    return app


app = create_app()