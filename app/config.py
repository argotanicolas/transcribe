from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    model_size: str = "small"  # renamed field — protected_namespaces suppressed below
    compute_type: str = "int8"
    device: str = "cpu"
    cpu_threads: int = 4       # ARM Ampere A1 tiene 4 OCPUs — usarlos todos
    num_workers: int = 2       # workers paralelos del modelo
    default_language: str = "es"  # idioma por defecto; pasar "auto" en la request para forzar detección
    max_file_size_mb: int = 1024
    temp_dir: str = "/tmp/transcribe"

    model_config = {"env_file": ".env", "protected_namespaces": ()}

settings = Settings()
