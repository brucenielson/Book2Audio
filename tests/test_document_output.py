import pytest
from pathlib import Path
from docling_parser import DoclingParser

TEST_DOCUMENTS = Path(__file__).parent / "test_documents"
TEST_CANONICAL = Path(__file__).parent / "test_canonical"


def compare_files(output_path: Path, canonical_path: Path) -> None:
    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    canonical_lines = canonical_path.read_text(encoding="utf-8").splitlines()

    differences = [
        f"Line {i + 1}:\n  expected: {canonical}\n  actual:   {actual}"
        for i, (actual, canonical) in enumerate(zip(output_lines, canonical_lines))
        if actual != canonical
    ]

    if len(output_lines) != len(canonical_lines):
        differences.append(
            f"Line count differs: expected {len(canonical_lines)}, got {len(output_lines)}"
        )

    if differences:
        first_five = "\n".join(differences[:5])
        pytest.fail(f"Output differs from canonical:\n{first_five}")


@pytest.fixture(scope="session")
def process_all_pdfs():
    """Process all PDFs in test_documents and generate debug text files."""
    pdf_files = list(TEST_DOCUMENTS.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in test_documents/")

    for pdf_file in pdf_files:
        parser = DoclingParser(
            source=pdf_file,
            meta_data={"source": pdf_file.name},
            min_paragraph_size=300,
        )
        parser.run(generate_text_file=True)

    return pdf_files


class TestDoclingParserOutput:
    def test_all_txt_files_have_canonical(self, process_all_pdfs):
        txt_files = list(TEST_DOCUMENTS.glob("*.txt"))
        missing = [f.name for f in txt_files if not (TEST_CANONICAL / f.name).exists()]
        if missing:
            pytest.fail(
                f"Missing canonical files for: {missing}\n"
                f"Copy the generated files from test_documents/ to test_canonical/ to create them."
            )

    def test_output_matches_canonical(self, process_all_pdfs):
        txt_files = list(TEST_DOCUMENTS.glob("*.txt"))
        failures = []
        for txt_file in txt_files:
            canonical_path = TEST_CANONICAL / txt_file.name
            if not canonical_path.exists():
                continue  # already caught by test_all_txt_files_have_canonical
            try:
                compare_files(txt_file, canonical_path)
            except pytest.fail.Exception as e:
                failures.append(str(e))

        if failures:
            pytest.fail("\n\n".join(failures))
