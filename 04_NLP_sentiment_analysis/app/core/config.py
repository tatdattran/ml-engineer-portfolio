from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NLP Sentiment Analysis"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_text_length: int = 2000
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
