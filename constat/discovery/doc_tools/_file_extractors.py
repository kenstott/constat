# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""File content extraction functions for PDF, DOCX, XLSX, PPTX."""


def _is_structured_data_format(doc_format: str) -> bool:
    """Check if format is structured data that shouldn't be semantically indexed."""
    # These formats are data to be queried, not text to be searched
    structured_formats = {"csv", "json", "jsonl", "parquet", "xml", "yaml", "yml"}
    return doc_format.lower() in structured_formats


def _extract_pdf_text(path) -> str:
    """Extract text content from a PDF file.

    Args:
        path: Path to the PDF file

    Returns:
        Extracted text content with page markers
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f"[Page {i}]\n{text.strip()}")

    return "\n\n".join(pages)


def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes.

    Args:
        pdf_bytes: Raw PDF file content

    Returns:
        Extracted text content with page markers
    """
    from io import BytesIO
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f"[Page {i}]\n{text.strip()}")

    return "\n\n".join(pages)


def _extract_docx_content(doc) -> str:
    """Extract text content from a python-docx Document object.

    Args:
        doc: python-docx Document instance

    Returns:
        Extracted text content with paragraph separation
    """
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # Check if it's a heading
            if para.style and para.style.name.startswith("Heading"):
                level = para.style.name.replace("Heading ", "")
                try:
                    level_num = int(level)
                    paragraphs.append(f"{'#' * level_num} {text}")
                except ValueError:
                    paragraphs.append(text)
            else:
                paragraphs.append(text)

    # Also extract text from tables
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            table_rows.append(" | ".join(cells))
        if table_rows:
            paragraphs.append("\n".join(table_rows))

    return "\n\n".join(paragraphs)


def _extract_docx_text(path) -> str:
    """Extract text content from a Word document.

    Args:
        path: Path to the .docx file

    Returns:
        Extracted text content with paragraph separation
    """
    from docx import Document

    return _extract_docx_content(Document(path))


def _extract_docx_text_from_bytes(docx_bytes: bytes) -> str:
    """Extract text content from Word document bytes.

    Args:
        docx_bytes: Raw .docx file content

    Returns:
        Extracted text content
    """
    from io import BytesIO
    from docx import Document

    return _extract_docx_content(Document(BytesIO(docx_bytes)))


def _extract_xlsx_content(wb) -> str:
    """Extract text content from an openpyxl Workbook object.

    Args:
        wb: openpyxl Workbook instance

    Returns:
        Extracted text content with sheet and cell markers
    """
    sheets = []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        rows = []

        for row in sheet.iter_rows():
            cells = []
            for cell in row:
                if cell.value is not None:
                    cells.append(str(cell.value))
                else:
                    cells.append("")
            # Only include rows that have some content
            if any(c.strip() for c in cells):
                rows.append(" | ".join(cells))

        if rows:
            sheets.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))

    return "\n\n".join(sheets)


def _extract_xlsx_text(path) -> str:
    """Extract text content from an Excel spreadsheet.

    Args:
        path: Path to the .xlsx file

    Returns:
        Extracted text content with sheet and cell markers
    """
    from openpyxl import load_workbook

    return _extract_xlsx_content(load_workbook(path, data_only=True))


def _extract_xlsx_text_from_bytes(xlsx_bytes: bytes) -> str:
    """Extract text content from Excel spreadsheet bytes.

    Args:
        xlsx_bytes: Raw .xlsx file content

    Returns:
        Extracted text content
    """
    from io import BytesIO
    from openpyxl import load_workbook

    return _extract_xlsx_content(load_workbook(BytesIO(xlsx_bytes), data_only=True))


def _extract_pptx_content(prs) -> str:
    """Extract text content from a python-pptx Presentation object.

    Args:
        prs: python-pptx Presentation instance

    Returns:
        Extracted text content with slide markers
    """
    slides = []

    for i, slide in enumerate(prs.slides, 1):
        slide_text = []

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())

            # Handle tables in slides
            if shape.has_table:
                table_rows = []
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    table_rows.append(" | ".join(cells))
                if table_rows:
                    slide_text.append("\n".join(table_rows))

        if slide_text:
            slides.append(f"[Slide {i}]\n" + "\n".join(slide_text))

    return "\n\n".join(slides)


def _extract_pptx_text(path) -> str:
    """Extract text content from a PowerPoint presentation.

    Args:
        path: Path to the .pptx file

    Returns:
        Extracted text content with slide markers
    """
    from pptx import Presentation

    return _extract_pptx_content(Presentation(path))


def _extract_pptx_text_from_bytes(pptx_bytes: bytes) -> str:
    """Extract text content from PowerPoint presentation bytes.

    Args:
        pptx_bytes: Raw .pptx file content

    Returns:
        Extracted text content
    """
    from io import BytesIO
    from pptx import Presentation

    return _extract_pptx_content(Presentation(BytesIO(pptx_bytes)))


def _detect_format(suffix: str) -> str:
    """Detect document format from file extension."""
    format_map = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".pptx": "pptx",
    }
    return format_map.get(suffix.lower(), "text")


def _detect_format_from_content_type(content_type: str) -> str:
    """Detect document format from HTTP content-type."""
    if "markdown" in content_type:
        return "markdown"
    if "html" in content_type:
        return "html"
    if "pdf" in content_type:
        return "pdf"
    return "text"
