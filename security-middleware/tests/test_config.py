import os

from app.config import Settings


def test_model_router_url_defaults_to_localhost():
    os.environ.pop("MODEL_ROUTER_BASE_URL", None)

    settings = Settings()

    assert settings.model_router_base_url == "http://127.0.0.1:8001"


def test_model_router_url_can_be_configured():
    os.environ["MODEL_ROUTER_BASE_URL"] = "http://model-router:8000"

    settings = Settings()

    assert settings.model_router_base_url == "http://model-router:8000"

    os.environ.pop("MODEL_ROUTER_BASE_URL", None)
