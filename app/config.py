from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    model_size: str = "small"
    compute_type: str = "int8"
    device: str = "cpu"
    max_file_size_mb: int = 500
    temp_dir: str = "/tmp/transcribe"

    class Config:
        env_file = ".env"

settings = Settings()
