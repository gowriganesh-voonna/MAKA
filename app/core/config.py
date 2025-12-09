# app/core/config.py
from pydantic_settings import BaseSettings   # <- changed import
from pydantic import Field
import os

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    # add other settings as needed

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
