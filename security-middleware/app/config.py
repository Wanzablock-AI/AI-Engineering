import os


class Settings:
    def __init__(self):
        self.model_router_base_url = os.getenv(
            "MODEL_ROUTER_BASE_URL",
            "http://127.0.0.1:8001",
        ).rstrip("/")
