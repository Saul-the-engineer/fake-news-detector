import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings

from api.routes import ROUTER

# Set up logging
logging.basicConfig(format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s", level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_dir: str = "models/trained/most_recent"

    class Config:
        env_file = ".env"


def create_app(settings: Settings) -> FastAPI:
    """Create a FastAPI app with configured middleware and routes."""
    app = FastAPI()

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.include_router(ROUTER)

    return app


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=8000)
