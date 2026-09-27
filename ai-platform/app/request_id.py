from uuid import uuid4


def generate_request_id() -> str:
    return f"req-{uuid4().hex}"
