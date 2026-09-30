"""Build real PDF and DOCX files for loader tests, without extra dependencies."""

from pathlib import Path

import docx
from pypdf import PdfWriter

FONT_OBJECT = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"


def write_text_pdf(path: Path, pages: list[str]) -> Path:
    path.write_bytes(serialize_pdf(pdf_objects_for_pages(pages)))
    return path


def write_blank_pdf(path: Path) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(path)
    return path


def write_docx(path: Path, paragraphs: list[str]) -> Path:
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(str(path))
    return path


def pdf_objects_for_pages(pages: list[str]) -> list[bytes]:
    # Object 1 catalog, 2 page tree, 3 font; page i is object 4+2i, its content 5+2i.
    page_ids = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    tree = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()
    objects = [catalog, tree, FONT_OBJECT]
    for page_id, text in zip(page_ids, pages, strict=True):
        objects += [pdf_page_object(page_id + 1), pdf_content_stream(text)]
    return objects


def pdf_page_object(content_id: int) -> bytes:
    page = "/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
    resources = "/Resources << /Font << /F1 3 0 R >> >>"
    return f"<< {page} {resources} /Contents {content_id} 0 R >>".encode()


def pdf_content_stream(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    return f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"


def serialize_pdf(objects: list[bytes]) -> bytes:
    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    return bytes(output) + pdf_xref_and_trailer(offsets, xref_start=len(output))


def pdf_xref_and_trailer(offsets: list[int], xref_start: int) -> bytes:
    entries = "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    size = len(offsets) + 1
    xref = f"xref\n0 {size}\n0000000000 65535 f \n{entries}"
    trailer = f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    return (xref + trailer).encode()
