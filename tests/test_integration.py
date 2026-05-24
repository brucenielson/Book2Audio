"""Integration tests — require a running LLM (Ollama).

Run with: pytest -m integration
"""

import re
import difflib
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from docling_core.types.doc.document import TextItem, DocItemLabel
from docling_core.types import DoclingDocument
from parsers.docling_parser import DoclingParser
from parsers.epub_parser import EpubParser
from text_cleaner import TextCleaner

from conftest import TEST_LLM_MODEL


# ── TextCleaner helpers ───────────────────────────────────────────────────────

def make_cleaner(model: str = TEST_LLM_MODEL, max_retries: int = 3) -> TextCleaner:
    return TextCleaner(model=model, max_retries=max_retries, temperature=0)


# ── DoclingParser helpers ─────────────────────────────────────────────────────

def _make_doc_item(spec, label: str, text: str, page_no: int = 1) -> MagicMock:
    item = MagicMock(spec=spec)
    item.label = label
    item.text = text
    prov = MagicMock()
    prov.page_no = page_no
    prov.bbox = MagicMock()
    prov.bbox.height = 10.0
    prov.bbox.t = 0.0
    prov.charspan = (0, 10)
    item.prov = [prov]
    return item


def _make_text_item(text: str, page_no: int = 1) -> MagicMock:
    return _make_doc_item(TextItem, DocItemLabel.TEXT.value, text, page_no)


def _make_parser(texts: list, cleaner: TextCleaner | None = None,
                 include_notes: bool = True) -> DoclingParser:
    doc = MagicMock(spec=DoclingDocument)
    doc.name = "test_doc"
    doc.texts = texts
    doc.pages = {}
    return DoclingParser(source=doc, meta_data={}, include_footnotes=include_notes,
                         llm_cleaner=cleaner, min_footnote_chars=100,
                         page_labels=None, skip_front_matter=False, skip_index=False)


# ── Document output helpers ───────────────────────────────────────────────────

TEST_DOCUMENTS_LLM = Path(__file__).parent / "test_documents_llm"
TEST_CANONICAL_LLM = Path(__file__).parent / "test_canonical_llm"

_SPELLING_VARIANT_THRESHOLD = 0.8


def _normalize(line: str) -> str:
    return re.sub(r'[^a-z0-9]', '', line.lower())


def _only_valid_spelling_variants(expected: str, actual: str) -> bool:
    expected_words = expected.lower().split()
    actual_words = actual.lower().split()
    opcodes = difflib.SequenceMatcher(None, expected_words, actual_words).get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        if tag == 'replace' and (i2 - i1) == (j2 - j1):
            for exp_word, act_word in zip(expected_words[i1:i2], actual_words[j1:j2]):
                exp_clean = re.sub(r'[^a-z]', '', exp_word)
                act_clean = re.sub(r'[^a-z]', '', act_word)
                if exp_clean == act_clean:
                    continue
                if difflib.SequenceMatcher(None, exp_clean, act_clean).ratio() < _SPELLING_VARIANT_THRESHOLD:
                    return False
        else:
            return False
    return True


def _compare_files(output_path: Path, canonical_path: Path) -> None:
    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    canonical_lines = canonical_path.read_text(encoding="utf-8").splitlines()
    differences = []
    for i in range(max(len(output_lines), len(canonical_lines))):
        if i >= len(output_lines):
            differences.append(f"Line {i + 1}:\n  expected: {canonical_lines[i]}\n  actual:   <missing>")
            continue
        if i >= len(canonical_lines):
            differences.append(f"Line {i + 1}:\n  expected: <missing>\n  actual:   {output_lines[i]}")
            continue
        exp, act = canonical_lines[i], output_lines[i]
        if _normalize(exp) == _normalize(act):
            continue
        if _only_valid_spelling_variants(exp, act):
            continue
        differences.append(f"Line {i + 1}:\n  expected: {exp}\n  actual:   {act}")
    if len(output_lines) != len(canonical_lines):
        differences.append(f"Line count differs: expected {len(canonical_lines)}, got {len(output_lines)}")
    if differences:
        pytest.fail(f"Output differs from canonical:\n{chr(10).join(differences[:5])}")


@pytest.fixture(scope="session")
def process_all_documents_with_cleaner():
    """Process all PDFs and EPUBs using the LLM cleaner and write output to test_documents_llm/."""
    pdf_files = list(TEST_DOCUMENTS_LLM.glob("*.pdf"))
    epub_files = list(TEST_DOCUMENTS_LLM.glob("*.epub"))

    if not pdf_files and not epub_files:
        pytest.skip("No PDF or EPUB files found in test_documents_llm/")

    TEST_DOCUMENTS_LLM.mkdir(exist_ok=True)
    cleaner = TextCleaner(model=TEST_LLM_MODEL, temperature=0)

    for pdf_path in pdf_files:
        parser = DoclingParser(source=pdf_path, meta_data={"source": pdf_path.name},
                               llm_cleaner=cleaner, start_page=3, end_page=4)
        docs, _ = parser.run(generate_text_file=True)
        output_path = TEST_DOCUMENTS_LLM / f"{pdf_path.stem}_processed_paragraphs.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc + "\n\n")

    for epub_path in epub_files:
        parser = EpubParser(source=epub_path, meta_data={"source": epub_path.name},
                            llm_cleaner=cleaner)
        docs, _ = parser.run()
        output_path = TEST_DOCUMENTS_LLM / f"{epub_path.stem}_processed_paragraphs.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc + "\n\n")

    return pdf_files + epub_files


# ── TestTextCleanerIntegration ────────────────────────────────────────────────

class TestTextCleanerIntegration:
    @pytest.mark.integration
    def test_real_llm_call_body(self) -> None:
        """Integration test — requires a running LLM."""
        cleaner = make_cleaner()
        paragraph = "This is a sample paragraph from a book about philosophy and rationality."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert "philosophy" in cleaned
        assert "rationality" in cleaned
        assert cleaned == paragraph

    @pytest.mark.integration
    def test_real_llm_call_footnote(self) -> None:
        """Integration test — requires a running LLM."""
        cleaner = make_cleaner()
        page_context = (
            "Others have found very similar defection rates in various minor religious sects.1\n\n"
            "1 This ignores the interesting question of whether the defectors have given up "
            "all the beliefs in the doctrines of the movement they have quit."
        )
        cleaned, classification = cleaner.clean(
            "1 This ignores the interesting question of whether the defectors have given up "
            "all the beliefs in the doctrines of the movement they have quit.",
            page_context=page_context
        )
        assert classification in ('footnote', 'drop')
        assert not cleaned.startswith("1 ")
        assert cleaned == ("This ignores the interesting question of whether the defectors have given up "
                           "all the beliefs in the doctrines of the movement they have quit.")

    @pytest.mark.integration
    def test_real_llm_call_drop(self) -> None:
        """Integration test — requires a running LLM."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "Chapter 1 ... 1\nChapter 2 ... 15\nChapter 3 ... 42"
        )
        assert classification == 'drop'

    @pytest.mark.integration
    def test_real_llm_call_body_unchanged(self) -> None:
        """Clean prose with no issues should be returned exactly as-is."""
        cleaner = make_cleaner()
        paragraph = "The French Revolution began in 1789 and fundamentally transformed the political landscape of Europe."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == paragraph

    @pytest.mark.integration
    def test_real_llm_call_ocr_word_break_fixed(self) -> None:
        """Mid-word line breaks introduced by OCR should be rejoined."""
        cleaner = make_cleaner()
        paragraph = "The development of mod- ern philosophy can be traced to the six- teenth century."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == "The development of modern philosophy can be traced to the sixteenth century."

    @pytest.mark.integration
    def test_real_llm_call_trailing_footnote_marker_stripped(self) -> None:
        """A trailing footnote number at the end of a body paragraph should be removed."""
        cleaner = make_cleaner()
        paragraph = "The movement grew rapidly throughout the nineteenth century, attracting followers from across the social spectrum. 4"
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == "The movement grew rapidly throughout the nineteenth century, attracting followers from across the social spectrum."

    @pytest.mark.integration
    def test_real_llm_call_footnote_identified_with_page_context(self) -> None:
        """A footnote paragraph should be identified and its leading number stripped when page context is provided."""
        cleaner = make_cleaner()
        page_context = (
            "The movement grew rapidly throughout the nineteenth century, "
            "attracting followers from across the social spectrum.4\n\n"
            "4 For full membership statistics by region, see Jones (1987), pp. 142-156."
        )
        paragraph = "4 For full membership statistics by region, see Jones (1987), pp. 142-156."
        cleaned, classification = cleaner.clean(paragraph, page_context=page_context)
        assert classification == 'footnote'
        assert cleaned.replace('–', '-') == "For full membership statistics by region, see Jones (1987), pp. 142-156."

    @pytest.mark.integration
    def test_real_llm_call_footnote_without_page_context(self) -> None:
        """Without page context the response should still be valid, even if classification varies."""
        cleaner = make_cleaner()
        paragraph = "4 For full membership statistics by region, see Jones (1987), pp. 142-156."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification in ('body', 'footnote', 'drop')
        assert isinstance(cleaned, str)

    @pytest.mark.integration
    def test_real_llm_call_drop_toc(self) -> None:
        """An obvious table of contents should be classified as drop."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "Introduction ... 1\nChapter One: The Early Years ... 15\n"
            "Chapter Two: The Middle Period ... 47\nConclusion ... 203"
        )
        assert classification == 'drop'


# ── TestDoclingParserIntegration ──────────────────────────────────────────────

class TestDoclingParserIntegration:
    @pytest.mark.integration
    def test_mislabelled_footnote_dropped_by_cleaner(self) -> None:
        """A footnote mislabeled as body text should be identified and dropped by the LLM cleaner."""
        texts = [
            _make_text_item(
                "Others have found very similar defection rates in various minor religious sects.1",
                page_no=1
            ),
            _make_text_item(
                "1 This ignores the interesting question of whether the defectors have given up "
                "all the beliefs in the doctrines of the movement they have quit.",
                page_no=1
            ),
        ]
        parser = _make_parser(texts, cleaner=TextCleaner(model=TEST_LLM_MODEL, temperature=0),
                               include_notes=False)
        docs, meta = parser.run()
        assert any("religious sects" in d for d in docs)
        assert all("This ignores the interesting question" not in d for d in docs)


# ── TestDocumentOutputWithCleaner ─────────────────────────────────────────────

class TestDocumentOutputWithLLMCleaner:
    @pytest.mark.integration
    def test_all_txt_files_have_canonical(self, process_all_documents_with_cleaner) -> None:
        txt_files = list(TEST_DOCUMENTS_LLM.glob("*.txt"))
        missing = [f.name for f in txt_files if not (TEST_CANONICAL_LLM / f.name).exists()]
        if missing:
            pytest.fail(
                f"Missing LLM canonical files for: {missing}\n"
                f"Copy the generated files from test_documents_llm/ to test_canonical_llm/ to create them."
            )

    @pytest.mark.integration
    def test_output_matches_canonical(self, process_all_documents_with_cleaner) -> None:
        txt_files = list(TEST_DOCUMENTS_LLM.glob("*.txt"))
        failures = []
        for txt_file in txt_files:
            canonical_path = TEST_CANONICAL_LLM / txt_file.name
            if not canonical_path.exists():
                continue
            try:
                _compare_files(txt_file, canonical_path)
            except pytest.fail.Exception as e:
                failures.append(f"{txt_file.name}:\n{e}")
        if failures:
            pytest.fail("\n\n".join(failures))
