from dataclasses import dataclass

from app.pdf_ingestion import Page


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    page_number: int
    text: str


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []

    for page in pages:
        text = page.text.strip()

        if not text:
            continue

        start = 0
        chunk_number = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{page.document}:"
                            f"{page.page_number}:"
                            f"{chunk_number}"
                        ),
                        document=page.document,
                        page_number=page.page_number,
                        text=chunk_text,
                    )
                )

            if end >= len(text):
                break

            start = end - overlap
            chunk_number += 1

    return chunks
