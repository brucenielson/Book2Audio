from unittest.mock import MagicMock
from parsers import EpubParser
from ebooklib import epub


# --- Fixtures ---

def make_epub_item(item_id: str, html: str) -> MagicMock:
    """Create a mock EpubHtml item with the given id and HTML content."""
    item = MagicMock()
    item.id = item_id
    item.get_body_content.return_value = html.encode('utf-8')
    return item


def make_epub_book(title: str, items: list) -> MagicMock:
    """Create a mock EpubBook with the given title and items."""
    book = MagicMock(spec=epub.EpubBook)
    book.title = title
    book.get_items_of_type.return_value = items
    return book


def make_parser(book: MagicMock | None = None,
                meta_data: dict | None = None,
                min_paragraph_size: int = 0,
                remove_footnotes: bool = True,
                sections_to_skip: list | None = None) -> EpubParser:
    """Create an EpubParser with a mock EpubBook."""
    if book is None:
        book = make_epub_book("Test Book", [])
    return EpubParser(
        source=book,
        meta_data=meta_data or {},
        min_paragraph_size=min_paragraph_size,
        remove_footnotes=remove_footnotes,
        sections_to_skip=sections_to_skip
    )


# --- TestParseSection ---

class TestParseSection:
    def test_basic_paragraph(self):
        parser = make_parser()
        html = "<p>This is a complete sentence.</p>"
        docs, meta = parser._parse_section(html, {})
        assert len(docs) == 1
        assert "This is a complete sentence." in docs[0]

    def test_empty_html(self):
        parser = make_parser()
        docs, meta = parser._parse_section("", {})
        assert docs == []
        assert meta == []

    def test_multiple_paragraphs_accumulated(self):
        parser = make_parser(min_paragraph_size=100)
        html = "<p>First sentence.</p><p>Second sentence.</p>"
        docs, meta = parser._parse_section(html, {})
        assert len(docs) == 1
        assert "First sentence." in docs[0]
        assert "Second sentence." in docs[0]

    def test_section_header_emitted_as_paragraph(self):
        parser = make_parser()
        html = "<h1>Chapter One</h1><p>Some content here.</p>"
        docs, meta = parser._parse_section(html, {})
        assert any("Chapter One" in d for d in docs)

    def test_section_header_flushes_accumulated(self):
        parser = make_parser(min_paragraph_size=100)
        html = "<p>Accumulated text.</p><h2>New Section</h2><p>New content.</p>"
        docs, meta = parser._parse_section(html, {})
        assert any("Accumulated text." in d for d in docs)
        assert any("New Section" in d for d in docs)

    def test_removes_footnotes(self):
        parser = make_parser(remove_footnotes=True)
        html = "<p>Main text.<sup>1</sup></p>"
        docs, meta = parser._parse_section(html, {})
        assert "1" not in docs[0]

    def test_keeps_footnotes_when_disabled(self):
        parser = make_parser(remove_footnotes=False)
        html = "<p>Main text.<sup>1</sup></p>"
        docs, meta = parser._parse_section(html, {})
        assert "1" in docs[0]

    def test_meta_contains_paragraph_number(self):
        parser = make_parser()
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        docs, meta = parser._parse_section(html, {})
        assert meta[0]["paragraph_#"] == "1"
        assert meta[1]["paragraph_#"] == "2"

    def test_chapter_title_in_meta(self):
        parser = make_parser()
        html = "<h1>Chapter One</h1><p>Some content here.</p>"
        docs, meta = parser._parse_section(html, {})
        content_meta = next((m for m in meta if "chapter_title" in m), None)
        assert content_meta is not None
        assert content_meta["chapter_title"] == "Chapter One"

    def test_section_name_in_meta(self):
        parser = make_parser()
        html = "<h2>My Section</h2><p>Some content here.</p>"
        docs, meta = parser._parse_section(html, {})
        content_meta = next((m for m in meta if "section_name" in m), None)
        assert content_meta is not None

    def test_page_number_in_meta(self):
        parser = make_parser()
        html = '<p><a id="page_5"></a>Content on page five.</p>'
        docs, meta = parser._parse_section(html, {})
        assert meta[0]["page_#"] == "5"


# --- TestRun ---

class TestRun:
    def test_basic_run(self):
        book = make_epub_book("Test Book", [
            make_epub_item("chapter1", "<p>First chapter content.</p>")
        ])
        parser = make_parser(book=book)
        docs, meta = parser.run()
        assert len(docs) == 1
        assert "First chapter content." in docs[0]

    def test_empty_book(self):
        book = make_epub_book("Empty Book", [])
        parser = make_parser(book=book)
        docs, meta = parser.run()
        assert docs == []
        assert meta == []

    def test_multiple_sections(self):
        book = make_epub_book("Test Book", [
            make_epub_item("chapter1", "<p>Chapter one content.</p>"),
            make_epub_item("chapter2", "<p>Chapter two content.</p>"),
        ])
        parser = make_parser(book=book)
        docs, meta = parser.run()
        assert any("Chapter one content." in d for d in docs)
        assert any("Chapter two content." in d for d in docs)

    def test_meta_contains_book_title(self):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>Some content.</p>")
        ])
        parser = make_parser(book=book)
        docs, meta = parser.run()
        assert meta[0]["book_title"] == "My Book"

    def test_meta_contains_item_id(self):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>Some content.</p>")
        ])
        parser = make_parser(book=book)
        docs, meta = parser.run()
        assert meta[0]["item_id"] == "chapter1"

    def test_meta_contains_item_number(self):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>Some content.</p>"),
            make_epub_item("chapter2", "<p>More content.</p>"),
        ])
        parser = make_parser(book=book)
        docs, meta = parser.run()
        assert meta[0]["item_#"] == "1"

    def test_skips_sections(self):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>Skipped content.</p>"),
            make_epub_item("chapter2", "<p>Included content.</p>"),
        ])
        parser = make_parser(book=book, sections_to_skip=["chapter1"])
        docs, meta = parser.run()
        assert all("Skipped content." not in d for d in docs)
        assert any("Included content." in d for d in docs)

    def test_sections_to_skip_multiple(self):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>Skipped one.</p>"),
            make_epub_item("chapter2", "<p>Skipped two.</p>"),
            make_epub_item("chapter3", "<p>Included content.</p>"),
        ])
        parser = make_parser(book=book, sections_to_skip=["chapter1", "chapter2"])
        docs, meta = parser.run()
        assert all("Skipped one." not in d for d in docs)
        assert all("Skipped two." not in d for d in docs)
        assert any("Included content." in d for d in docs)

    def test_generate_text_file_creates_files(self, tmp_path):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>Some content.</p>")
        ])
        epub_path = tmp_path / "test_book.epub"
        parser = EpubParser(source=book, meta_data={})
        parser._file_path = epub_path
        parser.run(generate_text_file=True)
        assert (tmp_path / "test_book_processed_paragraphs.txt").exists()
        assert (tmp_path / "test_book_processed_meta.txt").exists()

    def test_generate_text_file_content(self, tmp_path):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>Some content.</p>")
        ])
        epub_path = tmp_path / "test_book.epub"
        parser = EpubParser(source=book, meta_data={})
        parser._file_path = epub_path
        parser.run(generate_text_file=True)
        paragraphs = (tmp_path / "test_book_processed_paragraphs.txt").read_text(encoding="utf-8")
        assert "Some content." in paragraphs

    def test_accumulates_short_paragraphs(self):
        book = make_epub_book("My Book", [
            make_epub_item("chapter1", "<p>First sentence.</p><p>Second sentence.</p>")
        ])
        parser = make_parser(book=book, min_paragraph_size=100)
        docs, meta = parser.run()
        assert len(docs) == 1
        assert "First sentence." in docs[0]
        assert "Second sentence." in docs[0]
