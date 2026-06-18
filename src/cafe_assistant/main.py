from fastapi import FastAPI

from cafe_assistant.api.routes_chat import router as chat_router
from cafe_assistant.api.routes_consent import router as consent_router
from cafe_assistant.api.routes_observability import router as observability_router
from cafe_assistant.config import settings
from cafe_assistant.security.redaction import configure_redacted_logging


def create_app() -> FastAPI:
    configure_redacted_logging()
    app = FastAPI(title=settings.app_name)
    app.include_router(chat_router)
    app.include_router(consent_router)
    app.include_router(observability_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
