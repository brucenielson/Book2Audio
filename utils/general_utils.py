import csv
from typing import Optional, Dict, Any, Tuple
# from typing import List, Optional, Dict, Any, Union, Callable, Tuple, Set
from pathlib import Path
# import textwrap
# import re
# import generator_model as gen
# from ebooklib import ITEM_DOCUMENT, epub
# # noinspection PyPackageRequirements
# from haystack import Document, component
# from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever, PgvectorKeywordRetriever
# # noinspection PyPackageRequirements
# from haystack.components.preprocessors import DocumentSplitter
# from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
# # noinspection PyPackageRequirements
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# from neo4j_haystack import Neo4jEmbeddingRetriever
# from sentence_transformers import SentenceTransformer
# from html_parser import HTMLParser
# # noinspection PyPackageRequirements
# from haystack.dataclasses import ByteStream
# from pypdf import PdfReader, DocumentInformation
# import pymupdf4llm
# import pymupdf
# from transformers import AutoProcessor, BarkModel
# import sounddevice as sd
# from haystack.components.rankers import TransformersSimilarityRanker
# # noinspection PyPackageRequirements
# from haystack.utils import ComponentDevice, Device


def print_debug_results(results: Dict[str, Any],
                        include_outputs_from: Optional[set[str]] = None,
                        verbose: bool = True) -> None:
    level: int = 1
    if verbose and include_outputs_from is not None:
        # Exclude excess outputs
        results_filtered = {k: v for k, v in results.items() if k in include_outputs_from}
        if results_filtered:
            print()
            print("Debug Results:")
            # Call the recursive function to print the results hierarchically
            _print_hierarchy(results_filtered, level)


def _print_hierarchy(data: Dict[str, Any], level: int) -> None:
    for key, value in data.items():
        # Print the key with the corresponding level
        if level == 1:
            print()
        print(f"Level {level}: {key}")

        # Check if the value is a dictionary
        if isinstance(value, dict):
            _print_hierarchy(value, level + 1)
        # Check if the value is a list
        elif isinstance(value, list):
            for index, item in enumerate(value):
                print(f"Level {level + 1}: Item {index + 1}")  # Indicating it's an item in a list
                if isinstance(item, dict):
                    _print_hierarchy(item, level + 2)
                else:
                    print(item)  # Print the item directly
        else:
            # If the value is neither a dict nor a list, print it directly
            print(value)


def load_valid_pages(skip_file: str) -> Dict[str, Tuple[int, int]]:
    book_pages: Dict[str, Tuple[int, int]] = {}
    skip_file_path = Path(skip_file)

    if skip_file_path.exists():
        with open(skip_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader: csv.DictReader[str] = csv.DictReader(csvfile)
            row: dict[str, str]
            for row in reader:
                book_title: str = row['Book Title'].strip()
                start: str = row['Start'].strip()
                end: str = row['End'].strip()
                if book_title and start and end:
                    book_pages[book_title] = (int(start), int(end))

    return book_pages


# @component
# class EPubLoader:
#     def __init__(self, verbose: bool = False, skip_file: str = "sections_to_skip.csv") -> None:
#         self._verbose: bool = verbose
#         self._file_paths: List[str] = []
#         self._skip_file: str = skip_file
#         self._sections_to_skip: Dict[str, Set[str]] = {}
#
#     @component.output_types(html_pages=List[str], meta=List[Dict[str, str]])
#     def run(self, file_paths: Union[List[str], List[Path], str]) -> Dict[str, Any]:
#         # Handle not documents passed in
#         if len(file_paths) == 0:
#             return {"html_pages": [], "meta": []}
#         # Handle passing in a string with a path instead of a list of paths
#         if isinstance(file_paths, str):
#             file_paths = [file_paths]
#         # Handle passing in a list of Path objects instead of a list of strings
#         if isinstance(file_paths, list) and isinstance(file_paths[0], Path):
#             file_paths = [str(file_path) for file_path in file_paths]
#         # Verify that every single file path ends with .epub
#         if not all(file_path.lower().endswith('.epub') for file_path in file_paths):
#             raise ValueError("EpubLoader only accepts .epub files.")
#         self._file_paths = file_paths
#         self._sections_to_skip = self._load_sections_to_skip()
#         # Load the EPUB file
#         html_pages: List[str]
#         meta: List[Dict[str, str]]
#         html_pages, meta = self._load_files()
#         return {"html_pages": html_pages, "meta": meta}
#
#     def _load_files(self) -> Tuple[List[str], List[Dict[str, str]]]:
#         sources: List[str] = []
#         meta: List[Dict[str, str]] = []
#         for file_path in self._file_paths:
#             sources_temp: List[str]
#             meta_temp: List[Dict[str, str]]
#             sources_temp, meta_temp = self._load_epub(file_path)
#             sources.extend(sources_temp)
#             meta.extend(meta_temp)
#         return sources, meta
#
#     def _load_epub(self, file_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
#         book: epub.EpubBook = epub.read_epub(file_path)
#         self._print_verbose()
#         self._print_verbose(f"Loaded Book: {book.title}")
#         book_meta_data: Dict[str, str] = {
#             "book_title": book.title,
#             "file_path": file_path
#         }
#         i: int
#         item: epub.EpubHtml
#         html_pages: List[str] = []
#         meta_data: List[Dict[str, str]] = []
#         for i, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
#             if item.id not in self._sections_to_skip.get(book.title, set()):
#                 item_meta_data: Dict[str, str] = {
#                     "item_id": item.id
#                 }
#                 book_meta_data.update(item_meta_data)
#                 item_html: str = item.get_body_content().decode('utf-8')
#                 html_pages.append(item_html)
#                 meta_data.append(book_meta_data.copy())
#             else:
#                 self._print_verbose(f"Book: {book.title}; Section Id: {item.id}. User Skipped.")
#
#         return html_pages, meta_data
#
#     def _print_verbose(self, *args, **kwargs) -> None:
#         if self._verbose:
#             print(*args, **kwargs)
#
#     def _load_sections_to_skip(self) -> Dict[str, Set[str]]:
#         sections_to_skip: Dict[str, Set[str]] = {}
#         if os.path.isdir(self._file_paths[0]):
#             csv_path = Path(self._file_paths[0]) / self._skip_file
#         else:
#             # Get the directory of the file and then look for the csv file in that directory
#             csv_path = Path(self._file_paths[0]).parent / self._skip_file
#
#         if csv_path.exists():
#             with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
#                 reader: csv.DictReader[str] = csv.DictReader(csvfile)
#                 row: dict[str, str]
#                 for row in reader:
#                     book_title: str = row['Book Title'].strip()
#                     section_title: str = row['Section Title'].strip()
#                     if book_title and section_title:
#                         if book_title not in sections_to_skip:
#                             sections_to_skip[book_title] = set()
#                         sections_to_skip[book_title].add(section_title)
#
#             # Count total sections to skip across all books
#             skip_count: int = sum(len(sections) for _, sections in sections_to_skip.items())
#             self._print_verbose(f"Loaded {skip_count} sections to skip.")
#         else:
#             self._print_verbose("No sections_to_skip.csv file found. Processing all sections.")
#
#         return sections_to_skip
#
#
# @component
# class PdfLoader:
#     def __init__(self, verbose: bool = False, skip_file: str = "sections_to_skip.csv") -> None:
#         self._verbose: bool = verbose
#         self._file_paths: List[str] = []
#         self._skip_file: str = skip_file
#         self._sections_to_skip: Dict[str, Set[str]] = {}
#         self._converter: DocumentConverter = DocumentConverter()
#
#     @component.output_types(docling_docs=List[DoclingDocument], meta=List[Dict[str, str]])
#     def run(self, sources: Union[List[str], List[Path], str]) -> Dict[str, Any]:
#         file_paths: List[str] = sources
#         # Handle no documents passed in
#         if len(file_paths) == 0:
#             return {"docling_docs": [], "meta": []}
#         # Handle passing in a string with a path instead of a list of paths
#         if isinstance(file_paths, str):
#             file_paths = [file_paths]
#         # Handle passing in a list of Path objects instead of a list of strings
#         if isinstance(file_paths, list) and isinstance(file_paths[0], Path):
#             file_paths = [str(file_path) for file_path in file_paths]
#         # Verify that every single file path ends with .pdf
#         if not all(file_path.lower().endswith('.pdf') for file_path in file_paths):
#             raise ValueError("PdfLoader only accepts .pdf files.")
#         self._file_paths = file_paths
#         # self._sections_to_skip = self._load_sections_to_skip()
#         # Load the PDF file
#         docs: List[DoclingDocument]
#         meta: List[Dict[str, str]]
#         docs, meta = self._load_files()
#         return {"docling_docs": docs, "meta": meta}
#
#     def _load_files(self) -> Tuple[List[DoclingDocument], List[Dict[str, str]]]:
#         sources: List[DoclingDocument] = []
#         meta: List[Dict[str, str]] = []
#         for file_path in self._file_paths:
#             sources_temp: DoclingDocument
#             meta_temp: Dict[str, str]
#             sources_temp, meta_temp = self._load_pdf(file_path)
#             sources.append(sources_temp)
#             meta.append(meta_temp)
#         return sources, meta
#
#     def _load_pdf(self, file_path: str) -> Tuple[DoclingDocument, Dict[str, str]]:
#         # Check if already cached as a json
#         path = Path(file_path).with_suffix('.json')
#         book: DoclingDocument
#         if path.exists():
#             book = DoclingDocument.load_from_json(path)
#         else:
#             result: ConversionResult = self._converter.convert(file_path)
#             book = result.document
#             # Cache the book as a json
#             book.save_as_json(path)
#         self._print_verbose()
#         self._print_verbose(f"Loaded Book: {book.name}")
#         book_meta_data: Dict[str, str] = {
#             "book_title": book.name,
#             "file_path": file_path
#         }
#         # i: int
#         # item: epub.EpubHtml
#         # html_pages: List[str] = []
#         # meta_data: List[Dict[str, str]] = []
#         # for i, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
#         #     if item.id not in self._sections_to_skip.get(book.title, set()):
#         #         item_meta_data: Dict[str, str] = {
#         #             "item_id": item.id
#         #         }
#         #         book_meta_data.update(item_meta_data)
#         #         item_html: str = item.get_body_content().decode('utf-8')
#         #         html_pages.append(item_html)
#         #         meta_data.append(book_meta_data.copy())
#         #     else:
#         #         self._print_verbose(f"Book: {book.title}; Section Id: {item.id}. User Skipped.")
#
#         # return html_pages, meta_data
#         return book, book_meta_data
#
#     def _print_verbose(self, *args, **kwargs) -> None:
#         if self._verbose:
#             print(*args, **kwargs)
#
#
# @component
# class HTMLParserComponent:
#     def __init__(self, min_paragraph_size: int = 300, min_section_size: int = 1000, verbose: bool = False) -> None:
#         self._min_section_size: int = min_section_size
#         self._min_paragraph_size: int = min_paragraph_size
#         self._verbose: bool = verbose
#         self._sections_to_skip: Dict[str, Set[str]] = {}
#
#     @component.output_types(sources=List[ByteStream], meta=List[Dict[str, str]])
#     def run(self, html_pages: List[str], meta: List[Dict[str, str]]) -> Dict[str, Any]:
#         docs_list: List[ByteStream] = []
#         meta_list: List[Dict[str, str]] = []
#         included_sections: List[str] = []
#         missing_chapter_titles: List[str] = []
#         section_num: int = 1
#
#         for i, html_page in enumerate(html_pages):
#             page_meta_data: Dict[str, str] = meta[i]
#             parser: HTMLParser
#             item_id: str = page_meta_data.get("item_id", "").lower()
#             if item_id.startswith('notes'):
#                 parser = HTMLParser(html_page, page_meta_data, min_paragraph_size=self._min_paragraph_size * 2,
#                                     double_notes=False)  # If we're already doubling size, don't have parser do it too.
#             else:
#                 parser = HTMLParser(html_page, page_meta_data, min_paragraph_size=self._min_paragraph_size,
#                                     double_notes=True)
#
#             temp_docs: List[ByteStream]
#             temp_meta: List[Dict[str, str]]
#             temp_docs, temp_meta = parser.run()
#             item_id: str = page_meta_data.get("item_id", "")
#             book_title: str = page_meta_data.get("book_title", "")
#             if (parser.total_text_length() > self._min_section_size
#                     and item_id not in self._sections_to_skip.get(book_title, set())):
#                 self._print_verbose(f"Book: {book_title}; Section {section_num}. "
#                                     f"Chapter Title: {parser.chapter_title}. "
#                                     f"Length: {parser.total_text_length()}")
#                 # Add section number to metadata
#                 [meta.update({"item_#": str(section_num)}) for meta in temp_meta]
#                 docs_list.extend(temp_docs)
#                 meta_list.extend(temp_meta)
#                 included_sections.append(book_title + ", " + item_id)
#                 section_num += 1
#                 if parser.chapter_title is None or parser.chapter_title == "":
#                     missing_chapter_titles.append(book_title + ", " + item_id)
#             else:
#                 self._print_verbose(f"Book: {book_title}; Chapter Title: {parser.chapter_title}. "
#                                     f"Length: {parser.total_text_length()}. Skipped.")
#
#         if len(docs_list) > 0:
#             self._print_verbose(f"Sections included:")
#             for item in included_sections:
#                 self._print_verbose(item)
#             if missing_chapter_titles:
#                 self._print_verbose()
#                 self._print_verbose(f"Sections missing chapter titles:")
#                 for item in missing_chapter_titles:
#                     self._print_verbose(item)
#             self._print_verbose()
#         return {"sources": docs_list, "meta": meta_list}
#
#     def _print_verbose(self, *args, **kwargs) -> None:
#         if self._verbose:
#             print(*args, **kwargs)
#
#
# @component
# class DoclingParserComponent:
#     def __init__(self, min_paragraph_size: int = 300,
#                  min_section_size: int = 1000,
#                  skip_file: str = "documents/pdf_valid_pages.csv",
#                  verbose: bool = False) -> None:
#         self._min_section_size: int = min_section_size
#         self._min_paragraph_size: int = min_paragraph_size
#         self._verbose: bool = verbose
#         self._valid_pages: Dict[str, Tuple[int, int]] = {}
#         # Load pages to skip
#         self._valid_pages = load_valid_pages(skip_file)
#
#     @component.output_types(sources=List[ByteStream], meta=List[Dict[str, str]])
#     def run(self, sources: List[DoclingDocument], meta: List[Dict[str, str]]) -> Dict[str, Any]:
#         docs_list: List[ByteStream] = []
#         meta_list: List[Dict[str, str]] = []
#
#         for i, doc in enumerate(sources):
#             meta_data: Dict[str, str] = meta[i]
#             parser: DoclingParser
#             start_page: Optional[int] = None
#             end_page: Optional[int] = None
#             if doc.name in self._valid_pages:
#                 start_page, end_page = self._valid_pages[doc.name]
#             parser = DoclingParser(doc, meta_data,
#                                    min_paragraph_size=self._min_paragraph_size,
#                                    start_page=start_page,
#                                    end_page=end_page,
#                                    double_notes=True)
#             # Start here
#             temp_docs: List[ByteStream]
#             temp_meta: List[Dict[str, str]]
#             temp_docs, temp_meta = parser.run()
#             # item_id: str = meta_data.get("item_id", "")
#             book_title: str = meta_data.get("book_title", "")
#             # Unlike EPUB we don't have sections or chapters. So we don't need a total length.
#             # TODO: Add a way to skip pages instead.
#
#             self._print_verbose(f"Book: {book_title};")
#             docs_list.extend(temp_docs)
#             meta_list.extend(temp_meta)
#
#         return {"sources": docs_list, "meta": meta_list}
#
#     def _print_verbose(self, *args, **kwargs) -> None:
#         if self._verbose:
#             print(*args, **kwargs)


# def print_documents(documents: List[Document]) -> None:
#     ignore_keys: set = {'file_path', 'source_id'}
#     for i, doc in enumerate(documents, 1):
#         print(f"\nDocument {i}:")
#         print(f"Score: {doc.score}")
#
#         # Dynamically iterate over all keys in doc.meta, excluding 'file_path'
#         if hasattr(doc, 'meta') and doc.meta:
#             for key, value in doc.meta.items():
#                 if key.lower() in ignore_keys or key.startswith('_') or key.startswith('split'):
#                     continue
#                 # Print the key-value pair, wrapped at 80 characters
#                 print(textwrap.fill(f"{key.replace('_', ' ').title()}: {value}", width=80))
#
#         # Use text wrap to wrap the content at 80 characters
#         print(textwrap.fill(f"Content: {doc.content}", width=80))
#         print("-" * 50)
#
#
# def analyze_content(doc: Document, paragraph_num: int, title_line_max: int = 100) -> Dict[str, Optional[str]]:
#     result: Dict[str, Optional[Union[str, int]]] = {"chapter_number": None, "chapter_title": None,
#                                                     "cleaned_content": None}
#     # Split the content into lines
#     meta: Dict[str, str] = doc.meta
#     content: str = doc.content
#     section_id: str = meta.get("section_id", "").lower()
#     lines: List[str] = content.split("\n", 3)  # Only split into first two lines
#     first_line: str = lines[0].strip() if len(lines) > 0 else ""
#     second_line: str = lines[1].strip() if len(lines) > 1 else ""
#
#     # Check section title for the chapter number pattern if not already found
#     match = re.search(r'(?<!-)(?:chapter|ch)\D*(\d+)', section_id.lower())
#     if match and result["chapter_number"] is None:
#         result["chapter_number"] = int(match.group(1))  # Capture the chapter number
#
#     # Paragraph 1 is special - it may contain chapter number and title or DOI lines to remove
#     if paragraph_num == 1:
#         # Remove lines that start with "DOI:" on paragraph 1 - this is an unneeded line
#         if first_line.lower().startswith("doi:"):
#             content = content.replace(first_line, "", 1).strip()
#             result["cleaned_content"] = content
#             lines = content.split("\n", 3)  # Only split into first two lines
#             first_line = lines[0].strip() if len(lines) > 0 else ""
#             second_line = lines[1].strip() if len(lines) > 1 else ""
#
#         # Only analyze if the first line is under title_line_max characters
#         if len(first_line) < title_line_max:
#             # Check if the first line is a chapter number (an integer) - we prefer this over the section_id
#             if first_line.isdigit():
#                 result["chapter_number"] = int(first_line)
#                 # If first line is a lone chapter number, the second line is likely the chapter title
#                 if len(second_line) < title_line_max and result["chapter_title"] is None:
#                     result["chapter_title"] = second_line.title()
#             # Check if the first line is short enough to be a title
#             elif len(first_line) < title_line_max and first_line.isupper():
#                 result["chapter_title"] = first_line.title()
#
#     else:  # This is any other paragraph other than the first
#         # Check if the first line is a subsection title
#         # Patter is an integer followed by a period and then a title
#         if len(first_line) < title_line_max:
#             match = re.match(r'(\d+)\.\s*(.*)', first_line)
#             if match:
#                 result["subsection_num"] = int(match.group(1))
#                 result["subsection_title"] = match.group(2).title()
#
#     return result
