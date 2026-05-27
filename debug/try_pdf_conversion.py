"""Utility script for manually testing Docling PDF conversion."""

from __future__ import annotations

import pathlib

from docling.document_converter import DocumentConverter, ConversionResult
from docling_core.types import DoclingDocument


def docling_convert_pdf(source: str) -> DoclingDocument:
    """Convert a PDF file to a DoclingDocument and save it as JSON.

    Args:
        source: Path to the source PDF file.

    Returns:
        The converted DoclingDocument.
    """
    converter: DocumentConverter = DocumentConverter()
    print("Converting document using Docling...")
    result: ConversionResult = converter.convert(source)
    doc: DoclingDocument = result.document
    path = pathlib.Path("test_docling.json")
    doc.save_as_json(path)
    return doc


def docling_load_json(source: str) -> DoclingDocument:
    """Load a DoclingDocument from a JSON file.

    Args:
        source: Path to the JSON file to load.

    Returns:
        The loaded DoclingDocument.
    """
    path = pathlib.Path(source)
    doc: DoclingDocument = DoclingDocument.load_from_json(path)
    return doc


def docling_to_markdown(doc: DoclingDocument, output_path: str) -> str:
    """Export a DoclingDocument to a Markdown file.

    Args:
        doc: The DoclingDocument to export.
        output_path: Path to write the Markdown file.

    Returns:
        The exported Markdown string.
    """
    markdown: str = doc.export_to_markdown()
    path = pathlib.Path(output_path)
    path.write_text(markdown, encoding="utf-8")
    return markdown


def main() -> None:
    """Run a manual conversion test on a local document."""
    source = "documents/Realism and the Aim of Science -- Karl Popper -- 2017.pdf"  # document per local path or URL
    # doc = docling_convert_pdf(source)
    doc = docling_load_json("documents/test_docling.json")
    for text in doc.texts:
        print(text.label, text.text)
        print("\n")


if __name__ == "__main__":
    main()
