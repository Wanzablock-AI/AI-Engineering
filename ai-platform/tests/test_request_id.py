from app.request_id import generate_request_id


def test_generate_request_id_returns_non_empty_string():
    request_id = generate_request_id()

    assert isinstance(request_id, str)
    assert request_id


def test_generate_request_id_returns_unique_ids():
    first = generate_request_id()
    second = generate_request_id()

    assert first != second
