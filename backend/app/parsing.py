"""Text extraction for PDF, DOCX, JPG and PNG.

Digital PDFs use their text layer. Pages with no usable text layer (scans) are rendered
to images and sent through Tesseract OCR. Images are always OCR'd.
"""
from pathlib import Path

from PIL import Image


class ParsingError(Exception):
    pass


def extract_pages(path: Path) -> list[dict]:
    """Return a list of {"page": int, "text": str, "method": "text" | "ocr"}."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf_pages(path)
    if ext == ".docx":
        return _docx_pages(path)
    if ext in {".jpg", ".jpeg", ".png"}:
        return [{"page": 1, "text": _ocr(Image.open(path)), "method": "ocr"}]
    raise ParsingError(f"Unsupported file type: {ext}")


def _pdf_pages(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        method = "text"
        if len(text) < 20:  # no real text layer: treat as a scanned page
            text = _ocr(_render_pdf_page(path, i))
            method = "ocr"
        pages.append({"page": i + 1, "text": text, "method": method})
    return pages


def _render_pdf_page(path: Path, index: int) -> Image.Image:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return pdf[index].render(scale=2.0).to_pil()
    finally:
        pdf.close()


def _ocr(image: Image.Image) -> str:
    import pytesseract

    try:
        return pytesseract.image_to_string(image.convert("RGB")).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise ParsingError("Tesseract OCR is not installed. Install it to process scanned files.") from exc


def _docx_pages(path: Path) -> list[dict]:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [{"page": 1, "text": "\n".join(parts), "method": "text"}]
