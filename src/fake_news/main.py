"""Create a FastAPI app."""

import logging

import pydantic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fake_news.errors import (
    handle_broad_exceptions,
    handle_pydantic_validation_errors,
)
from fake_news.routes import ROUTER
from fake_news.settings import Settings


def create_app(settings: Settings | None) -> FastAPI:
    """Create a FastAPI app."""
    settings = settings or Settings()
    app = FastAPI()
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # This allows requests from any origin
        allow_credentials=True, # This allows cookies to be sent
        allow_methods=["*"], # This allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"], # This allows all headers to be sent
    )

    app.state.settings = settings
    app.include_router(ROUTER)
    app.add_exception_handler(
        exc_class_or_status_code=pydantic.ValidationError,
        handler=handle_pydantic_validation_errors,
    )
    app.middleware("http")(handle_broad_exceptions)

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app(settings=None)
    uvicorn.run(app, host="0.0.0.0", port=8000)
