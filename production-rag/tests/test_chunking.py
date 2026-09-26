from app.chunking import chunk_pages
from app.pdf_ingestion import Page


def test_chunk_pages_preserves_source_metadata():
    pages = [
        Page(
            document="test.pdf",
            page_number=3,
            text="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 3

    assert chunks[0].chunk_id == "test.pdf:3:0"
    assert chunks[0].document == "test.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].text == "ABCDEFGHIJ"

    assert chunks[1].text == "IJKLMNOPQR"
    assert chunks[2].text == "QRSTUVWXYZ"


def test_chunk_pages_rejects_invalid_configuration():
    pages = [
        Page(
            document="test.pdf",
            page_number=1,
            text="some text",
        )
    ]

    try:
        chunk_pages(pages, chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
