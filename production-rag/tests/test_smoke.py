def test_environment():
    import fastapi
    import pymupdf
    import pydantic
    import uvicorn

    assert fastapi.__version__
    assert pymupdf.__version__
    assert pydantic.__version__
    assert uvicorn.__version__
