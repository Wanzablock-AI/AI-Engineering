import httpx

from app.model_router import ModelRouterClient


def test_model_router_client_calls_generate():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:3b",
                "answer": "router-generated",
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )

    router = ModelRouterClient(
        base_url="http://model-router:8000",
        client=client,
    )

    answer = router.generate("Write Python code")

    assert answer == "router-generated"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://model-router:8000/generate"


def test_model_router_client_raises_for_http_error():
    def handler(request):
        return httpx.Response(
            502,
            json={"detail": "Model provider unavailable"},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )

    router = ModelRouterClient(
        base_url="http://model-router:8000",
        client=client,
    )

    try:
        router.generate("Explain Python lists")
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 502
    else:
        raise AssertionError("Expected HTTPStatusError")
