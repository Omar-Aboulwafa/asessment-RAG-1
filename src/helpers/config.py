from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: str = "pdf,docx,txt"
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    DB_CONNECTION_STRING: str
    OLLAMA_BASE_URL: str
    DEFAULT_LLM_MODEL: str
    DEFAULT_EMBEDDING_MODEL: str

    PHOENIX_COLLECTOR_ENDPOINT: str

    def get_allowed_types(self) -> List[str]:
        return [ext.strip().lower() for ext in self.FILE_ALLOWED_TYPES.split(',')]

def get_settings() -> Settings:
    return Settings()
