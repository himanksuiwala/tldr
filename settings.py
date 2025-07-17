from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    app_name: str = "tldr:"
    database_url: str = ""

    model_embedding:str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
    model_language:str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

settings = Settings()