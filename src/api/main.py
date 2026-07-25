"""FastAPI application factory."""

from fastapi import FastAPI

from .dependencies import get_settings
from .routes import router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    settings = get_settings()

    app = FastAPI(title=settings.app_title, version=settings.app_version)
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
