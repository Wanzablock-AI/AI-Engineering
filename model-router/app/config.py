import os


class Settings:
    def __init__(self):
        self.ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        ).rstrip("/")
