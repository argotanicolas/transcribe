from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    model_size: str = "small"  # renamed field — protected_namespaces suppressed below
    compute_type: str = "int8"
    device: str = "cpu"
    max_file_size_mb: int = 500
    temp_dir: str = "/tmp/transcribe"

    model_config = {"env_file": ".env", "protected_namespaces": ()}

settings = Settings()
