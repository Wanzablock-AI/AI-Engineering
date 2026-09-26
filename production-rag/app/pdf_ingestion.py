from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Page:
    document: str
    page_number: int
    text: str


def extract_pages(pdf_path: str | Path) -> list[Page]:
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    pages: list[Page] = []

    with pymupdf.open(path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()

            pages.append(
                Page(
                    document=path.name,
                    page_number=index + 1,
                    text=text,
                )
            )

    return pages
