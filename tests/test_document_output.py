import re
import difflib
import pytest
from pathlib import Path
from parsers.docling_parser import DoclingParser
from parsers.epub_parser import EpubParser
from text_cleaner import TextCleaner

TEST_DOCUMENTS = Path(__file__).parent / "test_documents"
TEST_CANONICAL = Path(__file__).parent / "test_canonical"
TEST_DOCUMENTS_LLM = Path(__file__).parent / "test_documents_llm"
TEST_CANONICAL_LLM = Path(__file__).parent / "test_canonical_llm"

_SPELLING_VARIANT_THRESHOLD = 0.8


def _normalize(line: str) -> str:
    return re.sub(r'[^a-z0-9]', '', line.lower())


def _only_valid_spelling_variants(expected: str, actual: str) -> bool:
    """Return True if all word differences between expected and actual are
    close spelling variants (e.g. American/British spelling differences like
    'initialize'/'initialise'). Uses character-level similarity — words must
    share at least 80% of their characters to qualify. This prevents unrelated
    dictionary words like 'text'/'footnote' from passing as variants."""
    expected_words = expected.lower().split()
    actual_words = actual.lower().split()

    opcodes = difflib.SequenceMatcher(None, expected_words, actual_words).get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        if tag == 'replace' and (i2 - i1) == (j2 - j1):
            # Same number of words substituted — each pair must be a close spelling variant
            for exp_word, act_word in zip(expected_words[i1:i2], actual_words[j1:j2]):
                exp_clean = re.sub(r'[^a-z]', '', exp_word)
                act_clean = re.sub(r'[^a-z]', '', act_word)
                if exp_clean == act_clean:
                    continue
                similarity = difflib.SequenceMatcher(None, exp_clean, act_clean).ratio()
                if similarity < _SPELLING_VARIANT_THRESHOLD:
                    return False
        else:
            # Insertions or deletions — not a spelling variant
            return False
    return True


def compare_files(output_path: Path, canonical_path: Path) -> None:
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
        differences.append(
            f"Line count differs: expected {len(canonical_lines)}, got {len(output_lines)}"
        )

    if differences:
        first_five = "\n".join(differences[:5])
        pytest.fail(f"Output differs from canonical:\n{first_five}")


@pytest.fixture(scope="session")
def process_all_documents():
    """Process all PDFs and EPUBs in test_documents and generate debug text files."""
    pdf_files = list(TEST_DOCUMENTS.glob("*.pdf"))
    epub_files = list(TEST_DOCUMENTS.glob("*.epub"))

    if not pdf_files and not epub_files:
        pytest.skip("No PDF or EPUB files found in test_documents/")

    for pdf_path in pdf_files:
        parser = DoclingParser(source=pdf_path, meta_data={"source": pdf_path.name})
        parser.run(generate_text_file=True)

    for epub_path in epub_files:
        parser = EpubParser(
            source=epub_path,
            meta_data={"source": epub_path.name}
        )
        parser.run(generate_text_file=True)

    return pdf_files + epub_files

@pytest.fixture(scope="session")
def process_all_documents_with_cleaner():
    """Process all PDFs and EPUBs using the LLM cleaner and write output to test_documents_llm/."""
    pdf_files = list(TEST_DOCUMENTS_LLM.glob("*.pdf"))
    epub_files = list(TEST_DOCUMENTS_LLM.glob("*.epub"))

    if not pdf_files and not epub_files:
        pytest.skip("No PDF or EPUB files found in test_documents/")

    TEST_DOCUMENTS_LLM.mkdir(exist_ok=True)
    cleaner = TextCleaner(temperature=0)

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


class TestDoclingParserOutput:
    @pytest.mark.canonical
    def test_all_txt_files_have_canonical(self, process_all_documents):
        txt_files = list(TEST_DOCUMENTS.glob("*.txt"))
        missing = [f.name for f in txt_files if not (TEST_CANONICAL / f.name).exists()]
        if missing:
            pytest.fail(
                f"Missing canonical files for: {missing}\n"
                f"Copy the generated files from test_documents/ to test_canonical/ to create them."
            )

    @pytest.mark.canonical
    def test_output_matches_canonical(self, process_all_documents):
        txt_files = list(TEST_DOCUMENTS.glob("*.txt"))
        failures = []
        for txt_file in txt_files:
            canonical_path = TEST_CANONICAL / txt_file.name
            if not canonical_path.exists():
                continue  # already caught by test_all_txt_files_have_canonical
            try:
                compare_files(txt_file, canonical_path)
            except pytest.fail.Exception as e:
                failures.append(f"{txt_file.name}:\n{e}")

        if failures:
            pytest.fail("\n\n".join(failures))


class TestDocumentOutputWithCleaner:
    @pytest.mark.integration
    def test_all_txt_files_have_canonical(self, process_all_documents_with_cleaner):
        txt_files = list(TEST_DOCUMENTS_LLM.glob("*.txt"))
        missing = [f.name for f in txt_files if not (TEST_CANONICAL_LLM / f.name).exists()]
        if missing:
            pytest.fail(
                f"Missing LLM canonical files for: {missing}\n"
                f"Copy the generated files from test_documents_llm/ to test_canonical_llm/ to create them."
            )

    @pytest.mark.integration
    def test_output_matches_canonical(self, process_all_documents_with_cleaner):
        txt_files = list(TEST_DOCUMENTS_LLM.glob("*.txt"))
        failures = []
        for txt_file in txt_files:
            canonical_path = TEST_CANONICAL_LLM / txt_file.name
            if not canonical_path.exists():
                continue  # already caught by test_all_txt_files_have_canonical
            try:
                compare_files(txt_file, canonical_path)
            except pytest.fail.Exception as e:
                failures.append(f"{txt_file.name}:\n{e}")

        if failures:
            pytest.fail("\n\n".join(failures))
