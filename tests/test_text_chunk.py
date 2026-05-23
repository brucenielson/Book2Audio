"""Tests for TextChunk and its subclasses."""

import pytest
from text_chunk import TextChunk, RawChunk, ParsedChunk


# --- is_section_header ---

class TestIsSectionHeader:
    @pytest.mark.parametrize("label", ["section_header", "title", "h1", "h2", "h3", "h4", "h5"])
    def test_is_true(self, label: str) -> None:
        assert TextChunk("text", label=label).is_section_header is True

    @pytest.mark.parametrize("label", ["text", "footnote", ""])
    def test_is_false(self, label: str) -> None:
        assert TextChunk("text", label=label).is_section_header is False


# --- is_footnote ---

class TestIsFootnote:
    @pytest.mark.parametrize("label", ["footnote"])
    def test_is_true(self, label: str) -> None:
        assert TextChunk("text", label=label).is_footnote is True

    @pytest.mark.parametrize("label", ["text", "section_header", ""])
    def test_is_false(self, label: str) -> None:
        assert TextChunk("text", label=label).is_footnote is False


# --- is_page_header ---

class TestIsPageHeader:
    @pytest.mark.parametrize("label", ["page_header"])
    def test_is_true(self, label: str) -> None:
        assert TextChunk("text", label=label).is_page_header is True

    @pytest.mark.parametrize("label", ["text", "page_footer", ""])
    def test_is_false(self, label: str) -> None:
        assert TextChunk("text", label=label).is_page_header is False


# --- is_page_footer ---

class TestIsPageFooter:
    @pytest.mark.parametrize("label", ["page_footer"])
    def test_is_true(self, label: str) -> None:
        assert TextChunk("text", label=label).is_page_footer is True

    @pytest.mark.parametrize("label", ["text", "page_header", ""])
    def test_is_false(self, label: str) -> None:
        assert TextChunk("text", label=label).is_page_footer is False


# --- is_body_text ---

class TestIsBodyText:
    @pytest.mark.parametrize("label", ["text", "list_item", "formula", "paragraph"])
    def test_is_true(self, label: str) -> None:
        assert TextChunk("text", label=label).is_body_text is True

    @pytest.mark.parametrize("label", ["section_header", "footnote", "page_header", ""])
    def test_is_false(self, label: str) -> None:
        assert TextChunk("text", label=label).is_body_text is False


# --- subclasses ---

class TestSubclasses:
    def test_raw_chunk_inherits_properties(self) -> None:
        chunk = RawChunk("text", label="section_header")
        assert chunk.is_section_header is True
        assert chunk.is_body_text is False

    def test_parsed_chunk_inherits_properties(self) -> None:
        chunk = ParsedChunk("note", label="footnote")
        assert chunk.is_footnote is True
        assert chunk.is_section_header is False

    def test_default_label_is_empty(self) -> None:
        chunk = TextChunk("text")
        assert chunk.label == ""
        assert chunk.is_body_text is False
        assert chunk.is_footnote is False

    def test_meta_defaults_to_empty_dict(self) -> None:
        chunk = TextChunk("text")
        assert chunk.meta == {}
