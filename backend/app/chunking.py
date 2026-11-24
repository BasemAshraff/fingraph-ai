"""Paragraph-aware chunking with overlap. Each chunk keeps its page number for citations."""
import re


def _split_long(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_pages(pages: list[dict], size: int = 800, overlap: int = 120) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        units: list[str] = []
        for para in re.split(r"\n\s*\n", page["text"]):
            para = " ".join(para.split())
            if not para:
                continue
            units.extend(_split_long(para, size) if len(para) > size else [para])

        buffer = ""
        for unit in units:
            if buffer and len(buffer) + len(unit) + 1 > size:
                chunks.append({"page": page["page"], "text": buffer})
                carry = buffer[-overlap:] if overlap else ""
                buffer = f"{carry} {unit}".strip() if len(carry) + len(unit) + 1 <= size else unit
            else:
                buffer = f"{buffer} {unit}".strip() if buffer else unit
        if buffer:
            chunks.append({"page": page["page"], "text": buffer})

    for index, chunk in enumerate(chunks):
        chunk["index"] = index
    return chunks
