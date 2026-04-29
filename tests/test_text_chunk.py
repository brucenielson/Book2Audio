"""Tests for TextChunk and its subclasses."""

from text_chunk import TextChunk, RawChunk, ParsedChunk


# --- is_section_header ---

class TestIsSectionHeader:
    def test_section_header_label(self) -> None:
        assert TextChunk("text", label="section_header").is_section_header is True

    def test_title_label(self) -> None:
        assert TextChunk("text", label="title").is_section_header is True

    def test_h1_label(self) -> None:
        assert TextChunk("text", label="h1").is_section_header is True

    def test_h2_label(self) -> None:
        assert TextChunk("text", label="h2").is_section_header is True

    def test_h3_label(self) -> None:
        assert TextChunk("text", label="h3").is_section_header is True

    def test_h4_label(self) -> None:
        assert TextChunk("text", label="h4").is_section_header is True

    def test_h5_label(self) -> None:
        assert TextChunk("text", label="h5").is_section_header is True

    def test_body_text_not_section_header(self) -> None:
        assert TextChunk("text", label="text").is_section_header is False

    def test_footnote_not_section_header(self) -> None:
        assert TextChunk("text", label="footnote").is_section_header is False

    def test_empty_label_not_section_header(self) -> None:
        assert TextChunk("text").is_section_header is False


# --- is_footnote ---

class TestIsFootnote:
    def test_footnote_label(self) -> None:
        assert TextChunk("text", label="footnote").is_footnote is True

    def test_body_text_not_footnote(self) -> None:
        assert TextChunk("text", label="text").is_footnote is False

    def test_section_header_not_footnote(self) -> None:
        assert TextChunk("text", label="section_header").is_footnote is False

    def test_empty_label_not_footnote(self) -> None:
        assert TextChunk("text").is_footnote is False


# --- is_page_header ---

class TestIsPageHeader:
    def test_page_header_label(self) -> None:
        assert TextChunk("text", label="page_header").is_page_header is True

    def test_body_text_not_page_header(self) -> None:
        assert TextChunk("text", label="text").is_page_header is False

    def test_page_footer_not_page_header(self) -> None:
        assert TextChunk("text", label="page_footer").is_page_header is False

    def test_empty_label_not_page_header(self) -> None:
        assert TextChunk("text").is_page_header is False


# --- is_page_footer ---

class TestIsPageFooter:
    def test_page_footer_label(self) -> None:
        assert TextChunk("text", label="page_footer").is_page_footer is True

    def test_body_text_not_page_footer(self) -> None:
        assert TextChunk("text", label="text").is_page_footer is False

    def test_page_header_not_page_footer(self) -> None:
        assert TextChunk("text", label="page_header").is_page_footer is False

    def test_empty_label_not_page_footer(self) -> None:
        assert TextChunk("text").is_page_footer is False


# --- is_body_text ---

class TestIsBodyText:
    def test_text_label(self) -> None:
        assert TextChunk("text", label="text").is_body_text is True

    def test_list_item_label(self) -> None:
        assert TextChunk("text", label="list_item").is_body_text is True

    def test_formula_label(self) -> None:
        assert TextChunk("text", label="formula").is_body_text is True

    def test_paragraph_label(self) -> None:
        assert TextChunk("text", label="paragraph").is_body_text is True

    def test_section_header_not_body_text(self) -> None:
        assert TextChunk("text", label="section_header").is_body_text is False

    def test_footnote_not_body_text(self) -> None:
        assert TextChunk("text", label="footnote").is_body_text is False

    def test_page_header_not_body_text(self) -> None:
        assert TextChunk("text", label="page_header").is_body_text is False

    def test_empty_label_not_body_text(self) -> None:
        assert TextChunk("text").is_body_text is False


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
