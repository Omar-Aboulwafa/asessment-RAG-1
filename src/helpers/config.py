from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    APP_NAME: str
    APP_VERSION: str
    
    FILE_ALLOWED_TYPES: str = "pdf,docx,txt"  # Changed to str
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int
    
    DB_CONNECTION_STRING: str = "postgresql://admin:admin@localhost:5400/rag"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_MODEL: str = "qwen3:0.6b-q4_K_M"
    DEFAULT_EMBEDDING_MODEL: str = "mxbai-embed-large:latest"
    
    def get_allowed_types(self) -> List[str]:
        """Get file types as a list"""
        return [ext.strip().lower() for ext in self.FILE_ALLOWED_TYPES.split(',')]

def get_settings():
    return Settings()