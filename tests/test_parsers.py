"""Tests for the WeKnora-style document parser registry.

The fixtures are built as real (minimal) OOXML and OCF containers, so the builtin parsers are
exercised against the same structures Word, Excel, PowerPoint, and EPUB produce. Every test runs
offline: no pandoc, no MarkItDown, no OCR engine, and no LLM endpoint is involved.
"""

from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest

from raglite._config import RAGLiteConfig
from raglite._markdown import document_to_markdown
from raglite._parsers import (
    BUILTIN_ENGINE,
    MULTIMODAL_ENGINE,
    BaseParser,
    CSVParser,
    DocxParser,
    EPUBParser,
    HtmlParser,
    MarkdownParser,
    MistralParser,
    MultimodalParser,
    ParsedImage,
    ParserEngineRegistry,
    PDFParser,
    PPTXParser,
    UnsupportedDocumentError,
    XLSXParser,
    default_registry,
    parse_document,
    parser_report,
    supported_file_types,
)

# ---------------------------------------------------------------------------
# Container fixtures.
# ---------------------------------------------------------------------------

_WORD_DOCUMENT = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Annual Report</w:t></w:r></w:p>
    <w:p><w:r><w:t>Revenue grew in every region.</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>First bullet</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Region</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Revenue</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>EMEA</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>12</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""

_SHEET_ONE = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
    <row><c r="A2" t="s"><v>2</v></c><c r="B2"><v>42</v></c></row>
  </sheetData>
</worksheet>
"""

_SHEET_TWO = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row><c r="A1" t="s"><v>3</v></c></row>
  </sheetData>
</worksheet>
"""

_SHARED_STRINGS = """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <si><t>Region</t></si><si><t>Revenue</t></si><si><t>EMEA</t></si><si><t>Notes</t></si>
</sst>
"""

_WORKBOOK = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Summary" sheetId="1" r:id="rId1"/>
    <sheet name="Detail" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>
"""

_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Target="worksheets/sheet2.xml"/>
</Relationships>
"""

_SLIDE_ONE = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:p><a:r><a:t>Quarterly results</a:t></a:r></a:p></p:txBody></p:sp>
    <p:sp><p:txBody><a:p><a:r><a:t>Revenue up 12%</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>
"""

_SLIDE_TWO = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:p><a:r><a:t>Next steps</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>
"""

_EPUB_CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

_EPUB_PACKAGE = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Field Notes</dc:title>
  </metadata>
  <manifest>
    <item id="chap1" href="chap1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chap2" href="chap2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="chap1"/><itemref idref="chap2"/></spine>
</package>
"""

_EPUB_CHAPTER_ONE = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
  <h1>Chapter One</h1><p>The instrument was calibrated at dawn.</p>
</body></html>
"""

_EPUB_CHAPTER_TWO = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
  <h1>Chapter Two</h1><p>The particle counts remained stable.</p>
</body></html>
"""

# A 1x1 red PNG, small enough to inline and realistic enough to be a real image file.
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c49444154789c63f8cfc0f0190000fb03010001f5e6d0f700000000"
    "49454e44ae426082"
)


def _write_docx(path: Path) -> Path:
    """Write a minimal .docx container."""
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        )
        archive.writestr("word/document.xml", _WORD_DOCUMENT)
        archive.writestr("word/media/image1.png", _PNG_BYTES)
    return path


def _write_xlsx(path: Path) -> Path:
    """Write a minimal two-sheet .xlsx container."""
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", _WORKBOOK)
        archive.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        archive.writestr("xl/sharedStrings.xml", _SHARED_STRINGS)
        archive.writestr("xl/worksheets/sheet1.xml", _SHEET_ONE)
        archive.writestr("xl/worksheets/sheet2.xml", _SHEET_TWO)
    return path


def _write_pptx(path: Path) -> Path:
    """Write a minimal two-slide .pptx container."""
    with ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", _SLIDE_ONE)
        archive.writestr("ppt/slides/slide2.xml", _SLIDE_TWO)
        archive.writestr("ppt/media/image1.png", _PNG_BYTES)
    return path


def _write_epub(path: Path) -> Path:
    """Write a minimal two-chapter .epub container."""
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", _EPUB_CONTAINER)
        archive.writestr("OEBPS/content.opf", _EPUB_PACKAGE)
        archive.writestr("OEBPS/chap1.xhtml", _EPUB_CHAPTER_ONE)
        archive.writestr("OEBPS/chap2.xhtml", _EPUB_CHAPTER_TWO)
        archive.writestr("OEBPS/cover.png", _PNG_BYTES)
    return path


# ---------------------------------------------------------------------------
# Registry routing.
# ---------------------------------------------------------------------------


def test_registry_routes_each_file_type_to_its_parser() -> None:
    """Test that the builtin engine covers the formats WeKnora's builtin registry covers."""
    registry = default_registry()
    expected = {
        "pdf": PDFParser,
        "md": MarkdownParser,
        "txt": MarkdownParser,
        "html": HtmlParser,
        "htm": HtmlParser,
        "csv": CSVParser,
        "tsv": CSVParser,
        "docx": DocxParser,
        "pptx": PPTXParser,
        "xlsx": XLSXParser,
        "epub": EPUBParser,
    }
    for file_type, parser_class in expected.items():
        assert registry.get_parser_class(BUILTIN_ENGINE, file_type) is parser_class


def test_registry_reports_the_file_types_it_covers() -> None:
    """Test that the builtin engine advertises at least the formats it parses."""
    file_types = supported_file_types(engine=BUILTIN_ENGINE)
    assert {"pdf", "md", "html", "csv", "docx", "pptx", "xlsx", "epub"} <= file_types


def test_registry_raises_for_an_unsupported_file_type() -> None:
    """Test that a file type no engine covers raises instead of silently reading bytes."""
    registry = default_registry()
    with pytest.raises(UnsupportedDocumentError, match="Unsupported file type"):
        registry.get_parser_class(BUILTIN_ENGINE, "exe")


def test_requested_engine_that_lacks_the_file_type_falls_back_to_builtin() -> None:
    """Test WeKnora's fallback: a requested engine that does not cover the type yields builtin."""
    registry = default_registry()
    # pandoc does not list "xlsx", so the builtin parser must win.
    assert registry.get_parser_class("pandoc", "xlsx") is XLSXParser


def test_unavailable_engine_falls_back_to_builtin() -> None:
    """Test that an engine whose dependency is missing is skipped rather than failing."""

    class UnavailableParser(BaseParser):
        FILE_TYPES: frozenset[str] = frozenset({"md"})

        @classmethod
        def is_available(cls) -> bool:
            return False

        def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> Any:
            error_message = "This parser must never run."
            raise AssertionError(error_message)

    registry = ParserEngineRegistry()
    registry.register(BUILTIN_ENGINE, [MarkdownParser])
    registry.register("broken", [UnavailableParser], check_available=UnavailableParser.is_available)
    assert registry.is_available("broken") is False
    assert registry.get_parser_class("broken", "md") is MarkdownParser


def test_parser_failure_falls_back_to_builtin(tmp_path: Any) -> None:
    """Test that an engine that fails at runtime degrades to the builtin engine."""
    from raglite._parsers import ParserFallbackError

    class FailingParser(BaseParser):
        ENGINE: str = "failing"
        FILE_TYPES: frozenset[str] = frozenset({"md"})

        def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> Any:
            error_message = "the external engine could not read this file"
            raise ParserFallbackError(error_message)

    doc_path = tmp_path / "note.md"
    doc_path.write_text("# Title\n\nBody text.", encoding="utf-8")
    registry = ParserEngineRegistry()
    registry.register(BUILTIN_ENGINE, [MarkdownParser])
    registry.register("failing", [FailingParser])
    parsed = registry.parse(doc_path, engine="failing", config=None)
    assert parsed.content == "# Title\n\nBody text."
    assert parsed.engine == BUILTIN_ENGINE


def test_parser_report_lists_engines_and_availability() -> None:
    """Test that the report explains which engines are installed and how to install the rest."""
    report = parser_report()
    assert BUILTIN_ENGINE in report
    assert report[BUILTIN_ENGINE]["available"] is True
    assert report["pandoc"]["hint"]
    assert not report["pandoc"]["available"] or report["pandoc"]["file_types"]


# ---------------------------------------------------------------------------
# Builtin parsing.
# ---------------------------------------------------------------------------


def test_parse_markdown_and_plain_text(tmp_path: Any) -> None:
    """Test that Markdown and plain text are read as-is."""
    doc_path = tmp_path / "notes.md"
    doc_path.write_text("# Heading\n\nBody.", encoding="utf-8")
    parsed = parse_document(doc_path)
    assert parsed.content == "# Heading\n\nBody."
    assert parsed.file_type == "md"


def test_parse_html_keeps_structure_and_images(tmp_path: Any) -> None:
    """Test that HTML conversion preserves the structures that matter for retrieval."""
    doc_path = tmp_path / "page.html"
    doc_path.write_text(
        "<html><head><title>Report</title></head><body>"
        "<h1>Results</h1><p>Some <a href='https://example.com'>link</a> text.</p>"
        "<ul><li>First</li><li>Second</li></ul>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "<img src='chart.png' alt='chart'/>"
        "<script>ignored()</script>"
        "</body></html>",
        encoding="utf-8",
    )
    parsed = parse_document(doc_path)
    assert parsed.metadata["title"] == "Report"
    assert "# Results" in parsed.content
    assert "[link](https://example.com)" in parsed.content
    assert "- First" in parsed.content
    assert "| A | B |" in parsed.content
    assert "| --- | --- |" in parsed.content
    assert "![chart](chart.png)" in parsed.content
    assert "ignored()" not in parsed.content
    assert parsed.images == (
        ParsedImage(name="chart.png", media_type="image/png", source="chart.png"),
    )


def test_parse_csv_and_tsv(tmp_path: Any) -> None:
    """Test that delimited files are converted to Markdown tables."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,value\nalpha,1\nbeta,2\n", encoding="utf-8")
    parsed = parse_document(csv_path)
    assert "| name | value |" in parsed.content
    assert "| alpha | 1 |" in parsed.content

    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("name\tvalue\nalpha\t1\n", encoding="utf-8")
    assert "| alpha | 1 |" in parse_document(tsv_path).content


def test_parse_docx_keeps_headings_lists_and_tables(tmp_path: Any) -> None:
    """Test that a .docx is parsed without requiring python-docx or pandoc."""
    parsed = parse_document(_write_docx(tmp_path / "report.docx"))
    assert "# Annual Report" in parsed.content
    assert "Revenue grew in every region." in parsed.content
    assert "- First bullet" in parsed.content
    assert "| Region | Revenue |" in parsed.content
    assert "| EMEA | 12 |" in parsed.content
    assert [image.name for image in parsed.images] == ["image1.png"]
    assert parsed.images[0].data == _PNG_BYTES


def test_parse_xlsx_renders_one_table_per_sheet(tmp_path: Any) -> None:
    """Test that a .xlsx is parsed sheet by sheet, resolving shared strings."""
    content = parse_document(_write_xlsx(tmp_path / "book.xlsx")).content
    assert "## Summary" in content
    assert "## Detail" in content
    assert "| Region | Revenue |" in content
    assert "| EMEA | 42 |" in content
    assert "| Notes |" in content


def test_parse_pptx_renders_one_section_per_slide(tmp_path: Any) -> None:
    """Test that a .pptx is parsed slide by slide."""
    parsed = parse_document(_write_pptx(tmp_path / "deck.pptx"))
    assert "## Slide 1" in parsed.content
    assert "- Quarterly results" in parsed.content
    assert "- Revenue up 12%" in parsed.content
    assert "## Slide 2" in parsed.content
    assert [image.name for image in parsed.images] == ["image1.png"]


def test_parse_epub_follows_the_spine(tmp_path: Any) -> None:
    """Test that an EPUB is parsed in spine order and carries its title and images."""
    parsed = parse_document(_write_epub(tmp_path / "book.epub"))
    assert parsed.metadata["title"] == "Field Notes"
    assert "# Chapter One" in parsed.content
    assert "The instrument was calibrated at dawn." in parsed.content
    assert parsed.content.index("Chapter One") < parsed.content.index("Chapter Two")
    assert any(image.name == "cover.png" for image in parsed.images)


def test_parse_pdf_uses_pdftext(tmp_path: Any) -> None:
    """Test that a PDF is parsed with pdftext and loses no heading structure."""
    doc_path = Path(__file__).parent / "specrel.pdf"
    parsed = parse_document(doc_path)
    assert parsed.file_type == "pdf"
    assert parsed.content.startswith("# ON THE ELECTRODYNAMICS OF MOVING")


def test_image_without_a_processor_raises(tmp_path: Any) -> None:
    """Test that an image is rejected instead of being decoded as text."""
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(_PNG_BYTES)
    with pytest.raises(UnsupportedDocumentError, match="multimodal"):
        parse_document(image_path)


def test_multimodal_and_mistral_parsers_need_their_processor(tmp_path: Any) -> None:
    """Test that the engines backed by a document processor require it to be configured."""
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(_PNG_BYTES)
    with pytest.raises(UnsupportedDocumentError, match="multimodal"):
        MultimodalParser().parse(image_path)
    with pytest.raises(UnsupportedDocumentError, match="Mistral"):
        MistralParser().parse(image_path)


def test_document_to_markdown_uses_the_registry(tmp_path: Any) -> None:
    """Test that the public conversion entry point routes through the registry."""
    doc_path = _write_docx(tmp_path / "report.docx")
    assert "# Annual Report" in document_to_markdown(doc_path)


def test_multimodal_engine_is_preferred_for_images() -> None:
    """Test that a configured multimodal processor selects the multimodal engine."""
    from raglite._config import MultimodalConfig
    from raglite._markdown import _preferred_parser_engine

    config = RAGLiteConfig(
        llm="gpt-4o-mini",
        embedder="text-embedding-3-small",
        reranker=None,
        document_processor=MultimodalConfig(),
    )
    assert _preferred_parser_engine(Path("scan.png"), config) == MULTIMODAL_ENGINE
    # A PDF is only handed to the multimodal processor when the processor accepts it.
    assert _preferred_parser_engine(Path("paper.pdf"), config) == MULTIMODAL_ENGINE
    # A Markdown file is not a multimodal document, so the builtin engine keeps it.
    plain = RAGLiteConfig(llm="gpt-4o-mini", embedder="text-embedding-3-small", reranker=None)
    assert _preferred_parser_engine(Path("notes.md"), plain) is None
