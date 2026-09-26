import pymupdf

from app.pdf_ingestion import extract_pages


def create_test_pdf(path):
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((72, 72), "Production RAG test document.")

    page = document.new_page()
    page.insert_text((72, 72), "Second page with citation metadata.")

    document.save(path)
    document.close()


def test_extract_pages(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    create_test_pdf(pdf_path)

    pages = extract_pages(pdf_path)

    assert len(pages) == 2

    assert pages[0].document == "test.pdf"
    assert pages[0].page_number == 1
    assert "Production RAG" in pages[0].text

    assert pages[1].page_number == 2
    assert "citation metadata" in pages[1].text
