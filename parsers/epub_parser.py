from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Iterator, Optional, Set
from bs4 import BeautifulSoup, Tag
from ebooklib import ITEM_DOCUMENT, epub
from utils.general_utils import enhance_title, clean_text
from text_chunk import RawChunk
from text_processor import TextProcessor
from parsers.base_parser import BaseParser


def get_header_level(paragraph: Tag) -> Optional[int]:
    """Return the level of the header (1 for h1, 2 for h2, etc.), or None if not a header."""
    if paragraph.name.startswith('h') and paragraph.name[1:].isdigit():
        return int(paragraph.name[1:])
    if hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs:
        section_headers: List[str] = ['pre-title1', 'h']
        for cls in paragraph.attrs['class']:
            if cls.lower() in section_headers:
                return 0
            elif cls.lower().startswith('h') and cls[1:].isdigit():
                return int(cls[1:])
    return None


def is_title(tag: Tag) -> bool:
    """Check if a tag is styled as a title."""
    # noinspection SpellCheckingInspection
    keywords: List[str] = ['title', 'chtitle', 'tochead', 'title1', 'h1_label']
    is_a_title: bool = (hasattr(tag, 'attrs') and 'class' in tag.attrs and
                        any(cls.lower().startswith(keyword) or cls.lower().endswith(keyword)
                            for cls in tag.attrs['class'] for keyword in keywords))
    return is_a_title


def is_header1_title(paragraph: Tag, h1_count: int) -> bool:
    """Check if a tag is an h1 being used as a chapter title."""
    header_level: Optional[int] = get_header_level(paragraph)
    if header_level == 1 and h1_count == 1:
        return True
    return False


def is_section_title(tag: Tag) -> bool:
    """Check if the tag is a title, heading, or chapter number."""
    if tag is None:
        return False
    header_lvl: Optional[int] = get_header_level(tag)
    return is_title(tag) or header_lvl is not None or is_chapter_number(tag)


def is_chapter_number(paragraph: Tag) -> bool:
    """Check if a tag represents a chapter number."""
    # noinspection SpellCheckingInspection
    chapter_classes = ['chno', 'ch-num']
    # noinspection SpellCheckingInspection
    return (hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs and
            any(cls in paragraph.attrs['class'] for cls in chapter_classes) and
            paragraph.text.isdigit())


def get_page_num(paragraph: Tag) -> Optional[str]:
    """Try to get a page number from a tag.

    Returns the page number as a string to accommodate Roman numerals,
    or None if no page number is found.
    """
    tags: List[Tag] = paragraph.find_all(
        lambda x: (x.name == 'a' or x.name == 'span') and x.get('id')
        and (x['id'].startswith('page_') or (x['id'].startswith('p') and x['id'][1:].isdigit()))
    )
    page_num: Optional[str] = None
    if tags:
        for tag in tags:
            page_id = tag.get('id')
            if page_id.startswith('page_'):
                page_num = page_id.split('_')[-1]
            elif page_id.startswith('p') and page_id[1:].isdigit():
                page_num = page_id[1:]
    if not page_num:
        page_id: str = paragraph.get('id')
        if page_id and page_id.startswith('p'):
            page_num = page_id[1:].split('-')[0]
            try:
                page_num = str(int(page_num))
            except ValueError:
                page_num = None
    return page_num


def is_sup_first_content(tag: Tag, sup_tag: Tag) -> bool:
    """Check if a sup tag is the first meaningful content in a tag."""
    for content in tag.contents:
        if isinstance(content, str) and not content.strip():
            continue
        in_first_content: bool = content == sup_tag or (isinstance(content, Tag) and sup_tag in content.descendants)
        return in_first_content and sup_tag.text.strip() == content.text.strip()
    return False


def recursive_yield_tags(tag: Tag, remove_footnotes: bool = False) -> Iterator[Tag]:
    """Recursively yield leaf tags that contain text content.

    Skips div tags and digs deeper into them. Optionally removes footnote
    superscript tags unless they appear as the first content in a paragraph.

    Args:
        tag: The root tag to traverse.
        remove_footnotes: If True, removes sup tags that are not the first content.

    Yields:
        Leaf tags containing text content.
    """
    invalid_children: List[str] = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
    if not tag.name == 'div' and tag.get_text(strip=True) and not tag.find(invalid_children):
        tag_copy: Tag = deepcopy(tag)
        for br in tag_copy.find_all('br'):
            br.insert_after(' ')
        if remove_footnotes:
            for fn in tag_copy.find_all('sup'):
                if not is_sup_first_content(tag_copy, fn):
                    fn.extract()
        yield tag_copy
    else:
        for child in tag.children:
            if isinstance(child, Tag):
                yield from recursive_yield_tags(child, remove_footnotes=remove_footnotes)


def get_chapter_info(tags: List[Tag],
                     h1_tags: List[Tag],
                     h2_tags: List[Tag],
                     h3_tags: List[Tag]) -> Tuple[str, int, str]:
    if not tags:
        return "", 0, ""

    chapter_title: str = ""
    h1_tags = [tag for tag in h1_tags if not is_chapter_number(tag) and not is_title(tag)]
    h1_tag_count: int = len(h1_tags)
    h2_tag_count: int = len(h2_tags)
    h3_tag_count: int = len(h3_tags)
    chapter_number: int = 0
    tags_to_delete: List[int] = []
    first_page_num: str = ""

    for i, tag in enumerate(tags):
        first_page_num = get_page_num(tag) or first_page_num
        if is_title(tag):
            tags_to_delete.append(i)
            title_text = enhance_title(tag.text)
            if chapter_title:
                chapter_title += ": " + title_text
            else:
                chapter_title = title_text
        elif is_chapter_number(tag):
            chapter_number = int(tag.text.strip())
        elif chapter_title == "" and tag.name != 'p':
            if h1_tag_count == 1 and get_header_level(tag) == 1:
                chapter_title = enhance_title(tag.text)
            elif h1_tag_count == 0:
                if h2_tag_count == 1 and get_header_level(tag) == 2:
                    chapter_title = enhance_title(tag.text)
                elif h3_tag_count == 1 and get_header_level(tag) == 3:
                    chapter_title = enhance_title(tag.text)
        elif tag.name == 'p' and not is_chapter_number(tag):
            if chapter_title or i > 2:
                break

    for i in sorted(tags_to_delete, reverse=True):
        del tags[i]

    return chapter_title, chapter_number, first_page_num


class EpubParser(BaseParser):
    """Parses an EPUB file into cleaned paragraphs suitable for text-to-speech.

    Reads each HTML section of the EPUB, extracts text using BeautifulSoup,
    runs it through the clean_text pipeline, and chunks it into paragraphs
    of at least min_paragraph_size characters.

    Attributes:
        _file_path: Path to the EPUB file.
        _meta_data: Base metadata to include with every paragraph.
        _min_paragraph_size: Minimum character count for a paragraph chunk.
        _remove_footnotes: If True, footnote superscripts are removed from text.
        _sections_to_skip: Dict mapping book titles to sets of section IDs to skip.
    """

    def __init__(self, source: str | Path | epub.EpubBook,
                 include_footnotes: bool = False,
                 meta_data: dict[str, str] | None = None,
                 min_paragraph_size: int = 0,
                 sections_to_skip: List[str] | None = None) -> None:
        """Initialise EpubParser.

        Args:
            source: Path to the EPUB file, or a pre-loaded EpubBook instance.
            include_footnotes: If True, footnote content is included in the
                               output alongside body text. Defaults to False.
            meta_data: Base metadata dict to include with every paragraph.
                       Defaults to None (empty metadata).
            min_paragraph_size: Minimum character count before a paragraph is
                                emitted. For audio output, 0 is a reasonable
                                default since short paragraphs are simply read
                                as brief pauses. Defaults to 0.
            sections_to_skip: Optional list of section IDs to skip. Use
                              load_sections_to_skip from general_utils to
                              load section IDs from a CSV file.
        """
        if isinstance(source, epub.EpubBook):
            self._book: epub.EpubBook = source
            self._file_path: Path | None = None
        else:
            self._file_path = Path(source)
            self._book = epub.read_epub(self._file_path) or {}

        # Note: include_footnotes is currently not implemented in EpubParser,
        # but we store it for potential future use and to maintain a consistent interface with DoclingParser.
        self._include_footnotes: bool = include_footnotes
        self._meta_data: dict[str, str] = meta_data
        self._min_paragraph_size: int = min_paragraph_size
        self._remove_footnotes: bool = True
        self._sections_to_skip: Dict[str, Set[str]] = {}
        if sections_to_skip:
            self._sections_to_skip[self._book.title] = set(sections_to_skip)

    def run(self, generate_text_file: bool = False) -> Tuple[List[str], List[Dict[str, str]]]:
        """Parse the EPUB and return paragraphs and metadata.

        Args:
            generate_text_file: If True, saves processed paragraph file
                                alongside the source EPUB.

        Returns:
            A tuple of (docs, meta) where docs is a list of paragraph strings
            and meta is a list of metadata dicts, one per paragraph.
        """
        book: epub.EpubBook = self._book
        print(f"Loaded Book: {book.title}")

        all_docs: List[str] = []
        all_meta: List[Dict[str, str]] = []
        section_num: int = 0

        item: epub.EpubHtml
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            if item.id in self._sections_to_skip.get(book.title, set()):
                print(f"Skipping section: {item.id}")
                continue

            section_num += 1
            item_html: str = item.get_body_content().decode('utf-8')
            section_meta: Dict[str, str] = {
                **self._meta_data,
                "book_title": book.title,
                "item_id": item.id,
                "item_#": str(section_num),
            }
            section_docs, section_meta_list = self._parse_section(item_html, section_meta)
            all_docs.extend(section_docs)
            all_meta.extend(section_meta_list)

        if generate_text_file:
            self._save_text_files(all_docs, all_meta)

        return all_docs, all_meta

    def _save_text_files(self, docs: List[str], meta: List[Dict[str, str]]) -> None:
        """Save processed paragraphs and metadata to text files alongside the source EPUB.

        Args:
            docs: The list of paragraph strings to save.
            meta: The list of metadata dicts, one per paragraph.
        """
        if self._file_path is None:
            raise ValueError("Cannot save text files when EpubBook was passed directly — no file path available.")
        base_path: Path = self._file_path.parent / self._file_path.stem

        with open(f"{base_path}_processed_paragraphs.txt", "w", encoding="utf-8") as f:
            for text in docs:
                f.write(text + "\n\n")

        with open(f"{base_path}_processed_meta.txt", "w", encoding="utf-8") as f:
            for text, m in zip(docs, meta):
                meta_str: str = " | ".join(f"{k}: {v}" for k, v in sorted(m.items()))
                f.write(f"[{meta_str}]\n")
                f.write(text + "\n\n")

    def _parse_section(self, html: str,
                       section_meta: Dict[str, str]) -> Tuple[List[str], List[Dict[str, str]]]:
        """Parse a single HTML section into paragraphs using TextProcessor.

        Args:
            html: The raw HTML content of the section.
            section_meta: Base metadata for this section.

        Returns:
            A tuple of (docs, meta) for this section.
        """
        soup: BeautifulSoup = BeautifulSoup(html, 'html.parser')
        tags: List[Tag] = list(recursive_yield_tags(soup, remove_footnotes=self._remove_footnotes))
        h1_tags: List[Tag] = soup.find_all('h1')
        h2_tags: List[Tag] = soup.find_all('h2')
        h3_tags: List[Tag] = soup.find_all('h3')

        chapter_title, chapter_number, page_num = get_chapter_info(tags, h1_tags, h2_tags, h3_tags)

        headers: Dict[int, str] = {}
        combine_headers: bool = False
        chunks: List[RawChunk] = []

        # Emit chapter title as its own chunk
        if chapter_title:
            chunks.append(RawChunk(
                text=chapter_title,
                meta={**section_meta, "chapter_title": chapter_title},
                label='section_header'
            ))

        for j, tag in enumerate(tags):
            # Update page number if this tag has one
            page_num = get_page_num(tag) or page_num

            # Skip chapter number tags
            if is_chapter_number(tag):
                continue

            # Handle header tags
            if get_header_level(tag) is not None:
                header_level: int = get_header_level(tag)
                header_text: str = enhance_title(tag.text)
                if header_level >= 6:
                    tag.name = 'p'
                else:
                    headers = {level: text for level, text in headers.items() if level <= header_level}
                    if header_text:
                        if not combine_headers or header_level not in headers:
                            headers[header_level] = header_text
                            combine_headers = True
                        else:
                            headers[header_level] = headers[header_level] + ": " + header_text
                            combine_headers = True

                        chunks.append(RawChunk(
                            text=header_text,
                            meta={**section_meta, "section_name": header_text},
                            label='section_header'
                        ))
                    continue

            combine_headers = False

            # Set chapter title from h0-level header if not already set
            if not chapter_title and headers and 0 in headers:
                chapter_title = headers[0]

            # p_str: str = clean_text(tag.get_text()) # TODO: This seems a the wrong place to clean_text. Should only be called in TextProcessor. This whole method seems strange to me.
            p_str: str = tag.get_text()
            if not p_str:
                continue

            # Build metadata for this chunk
            chunk_meta: Dict[str, str] = {**section_meta}
            if page_num:
                chunk_meta["page_#"] = page_num
            if chapter_title:
                chunk_meta["chapter_title"] = chapter_title
            if chapter_number:
                chunk_meta["chapter_#"] = str(chapter_number)
            if headers:
                top_header_level: int = min(headers.keys())
                for level, text in headers.items():
                    if level == top_header_level:
                        chunk_meta["section_name"] = text
                    else:
                        chunk_meta["subsection_name"] = (
                            chunk_meta.get("subsection_name", "") +
                            (": " + text if "subsection_name" in chunk_meta else text)
                        )

            chunks.append(RawChunk(text=p_str, meta=chunk_meta, label='text'))

        processor: TextProcessor = TextProcessor(min_paragraph_size=self._min_paragraph_size)
        parsed_chunks = processor.process(chunks)

        docs: List[str] = [chunk.text for chunk in parsed_chunks]
        meta: List[Dict[str, str]] = [chunk.meta for chunk in parsed_chunks]
        return docs, meta
