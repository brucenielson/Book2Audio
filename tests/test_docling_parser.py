"""Tests for the DoclingParser class."""

import pytest
from unittest.mock import MagicMock
from docling_core.types.doc.document import SectionHeaderItem, TextItem, DocItemLabel
from docling_core.types import DoclingDocument
from parsers.docling_parser import DoclingParser
from text_cleaner import TextCleaner


# --- Fixtures ---

def make_doc_item(spec, label: str, text: str, page_no: int = 1) -> MagicMock:
    """Create a mock DocItem with the given label, text, and page number."""
    item = MagicMock(spec=spec)
    item.label = label
    item.text = text
    prov = MagicMock()
    prov.page_no = page_no
    prov.bbox = MagicMock()
    prov.bbox.height = 10.0
    prov.charspan = (0, 10)
    item.prov = [prov]
    return item


def make_text_item(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock regular text item."""
    return make_doc_item(TextItem, DocItemLabel.TEXT.value, text, page_no)


def make_section_header(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock section header item."""
    return make_doc_item(SectionHeaderItem, DocItemLabel.SECTION_HEADER.value, text, page_no)


def make_footnote(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock footnote item."""
    return make_doc_item(TextItem, DocItemLabel.FOOTNOTE.value, text, page_no)


def make_page_header(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock page header item."""
    return make_doc_item(TextItem, DocItemLabel.PAGE_HEADER.value, text, page_no)


def make_page_footer(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock page footer item."""
    return make_doc_item(TextItem, DocItemLabel.PAGE_FOOTER.value, text, page_no)


def make_parser(texts: list,
                meta_data: dict | None = None,
                min_paragraph_size: int = 0,
                start_page: int | None = None,
                end_page: int | None = None,
                include_notes: bool = True,
                cleaner: TextCleaner | None = None) -> DoclingParser:
    """Create a DoclingParser with a mocked DoclingDocument."""
    doc = MagicMock(spec=DoclingDocument)
    doc.name = "test_doc"
    doc.texts = texts
    return DoclingParser(source=doc, meta_data=meta_data or {}, min_paragraph_size=min_paragraph_size,
                         start_page=start_page, end_page=end_page, include_footnotes=include_notes,
                         llm_cleaner=cleaner)


# --- TestGetProcessedTexts ---

class TestGetProcessedTexts:
    def test_separates_regular_and_footnotes(self) -> None:
        texts = [
            make_text_item("Regular text."),
            make_footnote("Footnote text."),
        ]
        parser = make_parser(texts)
        regular, notes = parser._get_processed_texts()
        assert len(regular) == 1
        assert len(notes) == 1

    def test_skips_too_short_items(self) -> None:
        texts = [
            make_text_item("Hi"),
            make_text_item("This is a longer sentence."),
        ]
        parser = make_parser(texts)
        regular, notes = parser._get_processed_texts()
        assert len(regular) == 1
        assert regular[0].text == "This is a longer sentence."

    def test_regular_texts_before_notes(self) -> None:
        texts = [
            make_footnote("Footnote."),
            make_text_item("Regular text."),
        ]
        parser = make_parser(texts)
        regular, notes = parser._get_processed_texts()
        assert regular[0].text == "Regular text."
        assert notes[0].text == "Footnote."

    def test_empty_document(self) -> None:
        parser = make_parser([])
        regular, notes = parser._get_processed_texts()
        assert regular == []
        assert notes == []


# --- TestRun ---

class TestRun:
    def test_basic_paragraph(self) -> None:
        texts = [make_text_item("This is a complete sentence.")]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert len(docs) == 1
        assert "This is a complete sentence." in docs[0]

    def test_meta_contains_expected_keys(self) -> None:
        texts = [make_text_item("This is a complete sentence.")]
        parser = make_parser(texts, meta_data={"source": "test"})
        docs, meta = parser.run()
        assert meta[0]["source"] == "test"
        assert "section_name" in meta[0]
        assert "page_#" in meta[0]
        assert "paragraph_#" in meta[0]

    def test_section_header_becomes_paragraph(self) -> None:
        texts = [make_section_header("Chapter One")]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert any("Chapter One" in d for d in docs)

    def test_section_header_flushes_accumulated_paragraph(self) -> None:
        texts = [
            make_text_item("First sentence without end"),
            make_section_header("Chapter Two"),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert any("First sentence without end" in d for d in docs)
        assert any("Chapter Two" in d for d in docs)

    def test_skips_page_header(self) -> None:
        texts = [
            make_page_header("Page Header"),
            make_text_item("Real content."),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert all("Page Header" not in d for d in docs)

    def test_skips_page_footer(self) -> None:
        texts = [
            make_text_item("Real content."),
            make_page_footer("Page Footer"),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert all("Page Footer" not in d for d in docs)

    def test_include_notes_true(self) -> None:
        texts = [
            make_text_item("Main text."),
            make_footnote("Footnote content."),
        ]
        parser = make_parser(texts, include_notes=True)
        docs, meta = parser.run()
        assert any("Footnote content." in d for d in docs)

    def test_include_notes_false(self) -> None:
        texts = [
            make_text_item("Main text."),
            make_footnote("Footnote content."),
        ]
        parser = make_parser(texts, include_notes=False)
        docs, meta = parser.run()
        assert all("Footnote content." not in d for d in docs)

    def test_start_page_filters_early_pages(self) -> None:
        texts = [
            make_text_item("Page one content.", page_no=1),
            make_text_item("Page two content.", page_no=2),
        ]
        parser = make_parser(texts, start_page=2)
        docs, meta = parser.run()
        assert all("Page one content." not in d for d in docs)
        assert any("Page two content." in d for d in docs)

    def test_end_page_filters_later_pages(self) -> None:
        texts = [
            make_text_item("Page one content.", page_no=1),
            make_text_item("Page two content.", page_no=2),
        ]
        parser = make_parser(texts, end_page=1)
        docs, meta = parser.run()
        assert any("Page one content." in d for d in docs)
        assert all("Page two content." not in d for d in docs)

    def test_page_range_inclusive(self) -> None:
        texts = [
            make_text_item("Page one.", page_no=1),
            make_text_item("Page two.", page_no=2),
            make_text_item("Page three.", page_no=3),
        ]
        parser = make_parser(texts, start_page=1, end_page=2)
        docs, meta = parser.run()
        assert any("Page one." in d for d in docs)
        assert any("Page two." in d for d in docs)
        assert all("Page three." not in d for d in docs)

    def test_accumulates_short_paragraphs(self) -> None:
        texts = [
            make_text_item("First sentence."),
            make_text_item("Second sentence."),
        ]
        parser = make_parser(texts, min_paragraph_size=100)
        docs, meta = parser.run()
        assert len(docs) == 1
        assert "First sentence." in docs[0]
        assert "Second sentence." in docs[0]

    def test_paragraph_number_in_meta(self) -> None:
        texts = [
            make_text_item("First paragraph."),
            make_text_item("Second paragraph."),
        ]
        parser = make_parser(texts, min_paragraph_size=0)
        docs, meta = parser.run()
        assert meta[0]["paragraph_#"] == "1"
        assert meta[1]["paragraph_#"] == "2"

    def test_section_name_in_meta(self) -> None:
        texts = [
            make_section_header("Chapter One"),
            make_text_item("Content here."),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        # The paragraph after the section header should have the section name
        content_meta = next(m for m in meta if m["section_name"] == "Chapter One")
        assert content_meta is not None

    def test_empty_document_returns_empty(self) -> None:
        parser = make_parser([])
        docs, meta = parser.run()
        assert docs == []
        assert meta == []


# --- TestIntegration ---

class TestIntegration:
    @pytest.mark.integration
    def test_mislabelled_footnote_dropped_by_cleaner(self) -> None:
        """A footnote mislabeled as body text should be identified and dropped by the LLM cleaner."""
        texts = [
            make_text_item(
                "Others have found very similar defection rates in various minor religious sects.1",
                page_no=1
            ),
            make_text_item(
                "1 This ignores the interesting question of whether the defectors have given up "
                "all the beliefs in the doctrines of the movement they have quit.",
                page_no=1
            ),
        ]
        parser = make_parser(texts, cleaner=TextCleaner(temperature=0), include_notes=False)
        docs, meta = parser.run()
        assert any("religious sects" in d for d in docs)
        assert all("This ignores the interesting question" not in d for d in docs)
