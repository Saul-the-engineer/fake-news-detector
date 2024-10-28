"""Settings for the files API."""

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """
    Settings for the files API.

    Pydantic BaseSettings docs: https://pydantic-settings.readthedocs.io/en/stable/api.html#pydantic_settings.BaseSettings
    FastAPI guide to managing settings: https://fastapi.tiangolo.com/advanced/settings/
    """

    s3_bucket_name: str = Field(...)
    model_config = SettingsConfigDict(
        case_sensitive=False,
    )
