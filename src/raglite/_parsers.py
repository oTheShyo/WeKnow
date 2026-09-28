"""Document parsing: a registry of format parsers with engine routing and graceful fallback.

WeKnora's `docreader` service keeps the conversion of "file on disk" to "text the indexer can
chop up" behind a single abstraction: a `ParserEngineRegistry` that maps a (engine, file type)
pair to a parser class, falls back to the builtin engine when the requested engine does not
support the file type, and reports why an engine is unavailable. RAGLite had the parsers but not
the abstraction: `document_to_markdown` branched on the file suffix, used pdftext for PDFs, and
handed everything else to pandoc — which meant that a missing pandoc installation silently
degraded a `.docx`, `.xlsx`, `.pptx`, or `.epub` file to `Path.read_text()`.

This module closes that gap with the same three ingredients WeKnora uses.

1. `ParsedDocument`, the single output shape every parser returns: Markdown `content`, the
   `images` carried by the document (so an OCR or vision stage can consume them, exactly like
   WeKnora's `Document{content, images}`), and provenance metadata.
2. `BaseParser` subclasses, grouped per engine. The builtin engine covers PDF, Markdown, plain
   text, HTML, CSV/TSV, and the OOXML and OCF container formats (`.docx`, `.pptx`, `.xlsx`,
   `.epub`) using only the standard library, so no optional dependency is required to parse them.
   Optional engines (`pandoc`, `markitdown`, `multimodal`, `mistral`) register themselves when
   their dependency is installed and report a hint when it is not.
3. `ParserEngineRegistry`, which resolves (engine, file type) to a parser, verifies availability
   before honouring a requested engine, and falls back to the builtin engine — both when the
   engine does not support the file type and when the engine's parser fails at runtime.

Nothing here embeds or splits: parsing ends at Markdown, and the existing sentence/chunklet/chunk
pipeline takes over from there.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from xml.etree import ElementTree as ET

from raglite._config import MistralOCRConfig, MultimodalConfig, RAGLiteConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

# Engine names. `builtin` is the fallback engine and must always be registered.
BUILTIN_ENGINE = "builtin"
PANDOC_ENGINE = "pandoc"
MARKITDOWN_ENGINE = "markitdown"
MULTIMODAL_ENGINE = "multimodal"
MISTRAL_ENGINE = "mistral"

# Upper bound on the images carried out of a container format (docx/pptx/epub). Parsing is about
# text; the images are a hook for a downstream OCR or vision stage, not a bulk export.
MAX_PARSED_IMAGES = 32

# Media type guesses for the images extracted from a container format.
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".emf": "image/x-emf",
    ".wmf": "image/x-wmf",
}

# OOXML / OCF namespaces.
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_PRESENTATION_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_RELATIONSHIP_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_OPF_NS = "{http://www.idpf.org/2007/opf}"
_DUBLIN_CORE_NS = "{http://purl.org/dc/elements/1.1/}"

_HEADING_STYLE_PATTERN = re.compile(r"heading\s*([1-6])", re.IGNORECASE)
_SLIDE_NAME_PATTERN = re.compile(r"ppt/slides/slide(\d+)\.xml$")
_COLUMN_REF_PATTERN = re.compile(r"([A-Z]+)")


class UnsupportedDocumentError(ValueError):
    """Raised when a parser cannot handle a document.

    The registry treats this as "try the builtin engine instead", and re-raises it only when the
    builtin engine itself cannot handle the document.
    """


class ParserFallbackError(RuntimeError):
    """Raised when an optional parser engine fails at runtime and the builtin engine should try."""


@dataclass(frozen=True)
class ParsedImage:
    """An image carried by a parsed document.

    `data` is populated for images stored inside a container format (docx, pptx, epub); images
    referenced by a URL, as in HTML, carry only `source`.
    """

    name: str
    media_type: str = ""
    data: bytes = b""
    source: str = ""


@dataclass(frozen=True)
class ParsedDocument:
    """The output of a document parser: Markdown content plus any images it carries."""

    content: str
    images: tuple[ParsedImage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    engine: str = BUILTIN_ENGINE
    file_type: str = ""


def file_type_of(doc_path: Path) -> str:
    """Return the lowercased file extension of `doc_path` without the leading dot."""
    return doc_path.suffix.lower().lstrip(".")


# ---------------------------------------------------------------------------
# Markdown helpers shared by the container and markup parsers.
# ---------------------------------------------------------------------------


def _markdown_table(rows: list[list[str]]) -> str:
    """Render rows as a GitHub Flavored Markdown table, using the first row as the header."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header, *body = padded

    def render(row: list[str]) -> str:
        cells = [cell.replace("|", "\\|").replace("\n", " ").strip() for cell in row]
        return f"| {' | '.join(cells)} |"

    lines = [render(header), f"| {' | '.join('---' for _ in header)} |"]
    lines.extend(render(row) for row in body)
    return "\n".join(lines) + "\n\n"


class _MarkdownHTMLParser(HTMLParser):
    """Convert HTML to GitHub Flavored Markdown with the standard library.

    The conversion is deliberately lossy in the same direction WeKnora's HTML parser is: structure
    that matters to retrieval (headings, lists, tables, links, image references) is preserved as
    Markdown, layout and scripting are dropped.
    """

    HEADING_LEVELS: ClassVar[dict[str, int]] = {f"h{level}": level for level in range(1, 7)}
    SKIPPED_TAGS: ClassVar[frozenset[str]] = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: list[ParsedImage] = []
        self.title: str = ""
        self._buffer: list[str] = []
        self._heading_level: int | None = None
        self._link_targets: list[str] = []
        self._skip_depth: int = 0
        self._preformatted: bool = False
        self._table_rows: list[list[str]] = []
        self._table_row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_title: bool = False

    def to_markdown(self) -> str:
        """Return the Markdown accumulated so far."""
        self._flush_inline()
        return "".join(self.parts).strip() + "\n"

    # -- Inline buffer ----------------------------------------------------

    def _flush_inline(self) -> str:
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        return text

    def handle_data(self, data: str) -> None:
        """Collect character data into the innermost open container."""
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if self._cell is not None:
            self._cell.append(data)
            return
        self._buffer.append(data if self._preformatted else re.sub(r"\s+", " ", data))

    # -- Tags -------------------------------------------------------------

    def handle_starttag(  # noqa: C901, PLR0912
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Open a Markdown container for `tag`."""
        if tag in self.SKIPPED_TAGS:
            self._skip_depth += 1
            return
        attributes = {name: (value or "") for name, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag in self.HEADING_LEVELS:
            self._buffer.clear()
            self._heading_level = self.HEADING_LEVELS[tag]
        elif tag in {"p", "div", "tr", "blockquote", "figcaption", "caption"}:
            self._buffer.clear()
        elif tag == "br":
            self._buffer.append("\n")
        elif tag in {"ul", "ol", "table", "dl"}:
            self._flush_inline()
        elif tag == "li":
            self._buffer.clear()
        elif tag == "pre":
            self._flush_inline()
            self._preformatted = True
            self._buffer.clear()
        elif tag == "a" and attributes.get("href"):
            self._link_targets.append(attributes["href"])
            self._buffer.append("[")
        elif tag == "img":
            self._emit_image(attributes)
        elif tag in {"th", "td"}:
            self._cell = []
            if self._table_row is None:
                self._table_row = []
        elif tag == "hr":
            self._flush_inline()
            self.parts.append("---\n\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle a self-closing tag such as `<img/>`."""
        self.handle_starttag(tag, attrs)
        if tag not in self.SKIPPED_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:  # noqa: C901, PLR0912
        """Close the Markdown container opened by `tag`."""
        if tag in self.SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag in self.HEADING_LEVELS:
            text = self._flush_inline()
            level = self._heading_level or 1
            self._heading_level = None
            if text:
                self.parts.append(f"{'#' * level} {text}\n\n")
        elif tag == "a":
            text = self._flush_inline()
            target = self._link_targets.pop() if self._link_targets else ""
            if text:
                self.parts.append(f"[{text}]({target})" if target else text)
        elif tag == "li":
            text = self._flush_inline()
            if text:
                self.parts.append(f"- {text}\n")
        elif tag in {"ul", "ol", "dl"}:
            self.parts.append("\n")
        elif tag == "pre":
            text = "".join(self._buffer).strip()
            self._buffer.clear()
            self._preformatted = False
            if text:
                self.parts.append(f"```\n{text}\n```\n\n")
        elif tag in {"p", "div", "blockquote", "figcaption", "caption"}:
            text = self._flush_inline()
            if text:
                self.parts.append(f"{text}\n\n")
        elif tag in {"th", "td"}:
            if self._cell is not None and self._table_row is not None:
                self._table_row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr":
            if self._table_row is not None:
                self._table_rows.append(self._table_row)
                self._table_row = None
        elif tag == "table":
            self.parts.append(_markdown_table(self._table_rows))
            self._table_rows = []

    def _emit_image(self, attributes: Mapping[str, str]) -> None:
        """Record an image reference as a Markdown image and as a `ParsedImage`."""
        source = attributes.get("src", "")
        alt = attributes.get("alt", "")
        name = Path(source).name or "image"
        self.parts.append(f"![{alt}]({source})")
        self.images.append(
            ParsedImage(
                name=name,
                media_type=_IMAGE_MEDIA_TYPES.get(Path(source).suffix.lower(), ""),
                source=source,
            )
        )


def html_to_markdown(html_content: str) -> tuple[str, tuple[ParsedImage, ...], str]:
    """Convert an HTML document to Markdown, returning (content, images, title)."""
    parser = _MarkdownHTMLParser()
    parser.feed(html_content)
    parser.close()
    return parser.to_markdown(), tuple(parser.images), parser.title.strip()


# ---------------------------------------------------------------------------
# Container formats (docx, pptx, xlsx, epub) read with the standard library.
# ---------------------------------------------------------------------------


def _read_xml(archive: zipfile.ZipFile, member: str) -> ET.Element:
    """Parse an XML member of a zip container."""
    return ET.fromstring(archive.read(member))  # noqa: S314


def _container_images(archive: zipfile.ZipFile, prefix: str) -> tuple[ParsedImage, ...]:
    """Collect the image files stored under `prefix` in a zip container."""
    images: list[ParsedImage] = []
    for name in sorted(archive.namelist()):
        if not name.startswith(prefix):
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in _IMAGE_MEDIA_TYPES:
            continue
        images.append(
            ParsedImage(
                name=Path(name).name,
                media_type=_IMAGE_MEDIA_TYPES[suffix],
                data=archive.read(name),
            )
        )
        if len(images) >= MAX_PARSED_IMAGES:
            break
    return tuple(images)


def _word_paragraph_text(paragraph: ET.Element) -> str:
    """Return the text of a WordprocessingML paragraph."""
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_WORD_NS}t":
            parts.append(node.text or "")
        elif node.tag == f"{_WORD_NS}tab":
            parts.append(" ")
        elif node.tag == f"{_WORD_NS}br":
            parts.append("\n")
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def _word_paragraph_style(paragraph: ET.Element) -> str:
    """Return the style id of a WordprocessingML paragraph."""
    properties = paragraph.find(f"{_WORD_NS}pPr")
    if properties is None:
        return ""
    style = properties.find(f"{_WORD_NS}pStyle")
    if style is None:
        return ""
    return style.get(f"{_WORD_NS}val", "")


def _word_heading_level(style: str) -> int | None:
    """Return the heading level implied by a WordprocessingML style id."""
    if not style:
        return None
    if match := _HEADING_STYLE_PATTERN.search(style):
        return int(match.group(1))
    if style.lower() in {"title", "subtitle"}:
        return 1
    return None


def _word_is_list_item(paragraph: ET.Element) -> bool:
    """Return whether a WordprocessingML paragraph is a numbered or bulleted list item."""
    properties = paragraph.find(f"{_WORD_NS}pPr")
    return properties is not None and properties.find(f"{_WORD_NS}numPr") is not None


def _word_table_to_markdown(table: ET.Element) -> str:
    """Render a WordprocessingML table as Markdown."""
    rows: list[list[str]] = []
    for row in table.findall(f"{_WORD_NS}tr"):
        cells = [
            " ".join(_word_paragraph_text(paragraph) for paragraph in cell.findall(f"{_WORD_NS}p"))
            for cell in row.findall(f"{_WORD_NS}tc")
        ]
        rows.append([cell.strip() for cell in cells])
    return _markdown_table([row for row in rows if any(row)])


def docx_to_markdown(archive: zipfile.ZipFile) -> str:
    """Convert the `word/document.xml` of a .docx container to Markdown."""
    body = _read_xml(archive, "word/document.xml").find(f"{_WORD_NS}body")
    if body is None:
        return ""
    parts: list[str] = []
    for child in body:
        if child.tag == f"{_WORD_NS}p":
            text = _word_paragraph_text(child)
            if not text:
                continue
            if level := _word_heading_level(_word_paragraph_style(child)):
                parts.append(f"{'#' * level} {text}\n\n")
            elif _word_is_list_item(child):
                parts.append(f"- {text}\n")
            else:
                parts.append(f"{text}\n\n")
        elif child.tag == f"{_WORD_NS}tbl":
            parts.append(_word_table_to_markdown(child))
    return "".join(parts).strip() + "\n"


def _slide_number(name: str) -> int:
    """Return the slide number of a `ppt/slides/slideN.xml` member name."""
    match = _SLIDE_NAME_PATTERN.search(name)
    return int(match.group(1)) if match else 0


def _sheet_number(name: str) -> int:
    """Return the sheet number of an `xl/worksheets/sheetN.xml` member name."""
    match = re.search(r"(\d+)", Path(name).stem)
    return int(match.group(1)) if match else 0


def pptx_to_markdown(archive: zipfile.ZipFile) -> str:
    """Convert the slides of a .pptx container to Markdown, one section per slide."""
    slide_names = sorted(
        (name for name in archive.namelist() if _SLIDE_NAME_PATTERN.search(name)), key=_slide_number
    )
    parts: list[str] = []
    for index, name in enumerate(slide_names, start=1):
        root = _read_xml(archive, name)
        lines: list[str] = []
        for paragraph in root.iter(f"{_DRAWING_NS}p"):
            text = "".join(node.text or "" for node in paragraph.iter(f"{_DRAWING_NS}t")).strip()
            if text:
                lines.append(f"- {text}")
        if not lines:
            continue
        parts.append(f"## Slide {index}\n\n" + "\n".join(lines) + "\n\n")
    return "".join(parts).strip() + "\n"


def _spreadsheet_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Return the shared strings of a .xlsx container."""
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = _read_xml(archive, "xl/sharedStrings.xml")
    return [
        "".join(node.text or "" for node in item.iter(f"{_SPREADSHEET_NS}t"))
        for item in root.findall(f"{_SPREADSHEET_NS}si")
    ]


def _spreadsheet_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return the (name, member) pairs of the sheets in a .xlsx container, in workbook order."""
    names = archive.namelist()
    members = sorted(
        (name for name in names if re.match(r"xl/worksheets/sheet\d+\.xml$", name)),
        key=_sheet_number,
    )
    if "xl/workbook.xml" not in names:
        return [(Path(member).stem, member) for member in members]
    target_by_id: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in names:
        relationships = _read_xml(archive, "xl/_rels/workbook.xml.rels")
        for relationship in relationships:
            target_by_id[relationship.get("Id", "")] = relationship.get("Target", "")
    workbook = _read_xml(archive, "xl/workbook.xml")
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.iter(f"{_SPREADSHEET_NS}sheet"):
        target = target_by_id.get(sheet.get(f"{_RELATIONSHIP_NS}id", ""), "")
        member = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
        if member in names:
            sheets.append((sheet.get("name", Path(member).stem), member))
    return sheets or [(Path(member).stem, member) for member in members]


def _column_index(reference: str) -> int:
    """Return the zero-based column index of a spreadsheet cell reference such as `B12`."""
    match = _COLUMN_REF_PATTERN.search(reference)
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def xlsx_to_markdown(archive: zipfile.ZipFile) -> str:
    """Convert the sheets of a .xlsx container to Markdown tables."""
    shared_strings = _spreadsheet_shared_strings(archive)
    parts: list[str] = []
    for sheet_name, member in _spreadsheet_sheets(archive):
        rows: list[list[str]] = []
        for row in _read_xml(archive, member).iter(f"{_SPREADSHEET_NS}row"):
            cells: list[str] = []
            for cell in row.findall(f"{_SPREADSHEET_NS}c"):
                value_node = cell.find(f"{_SPREADSHEET_NS}v")
                cell_type = cell.get("t", "")
                if cell_type == "s" and value_node is not None and value_node.text:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(f"{_SPREADSHEET_NS}t"))
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                index = _column_index(cell.get("r", ""))
                cells.extend([""] * max(0, index - len(cells)))
                cells.append(value.strip())
            rows.append(cells)
        rows = [row for row in rows if any(cell for cell in row)]
        if not rows:
            continue
        parts.append(f"## {sheet_name}\n\n" + _markdown_table(rows))
    return "".join(parts).strip() + "\n"


def _epub_content_documents(archive: zipfile.ZipFile) -> tuple[list[str], str]:
    """Return the (ordered content members, title) of an EPUB container."""
    container = _read_xml(archive, "META-INF/container.xml")
    rootfile = container.find(f"{_CONTAINER_NS}rootfiles/{_CONTAINER_NS}rootfile")
    if rootfile is None:
        return [], ""
    rootfile_path = rootfile.get("full-path", "")
    package = _read_xml(archive, rootfile_path)
    base = Path(rootfile_path).parent
    href_by_id: dict[str, str] = {}
    for item in package.iter(f"{_OPF_NS}item"):
        href_by_id[item.get("id", "")] = item.get("href", "")
    members: list[str] = []
    for itemref in package.iter(f"{_OPF_NS}itemref"):
        href = href_by_id.get(itemref.get("idref", ""), "")
        member = (base / href).as_posix() if href else ""
        if member in archive.namelist():
            members.append(member)
    title_node = package.find(f"{_OPF_NS}metadata/{_DUBLIN_CORE_NS}title")
    title = (title_node.text or "") if title_node is not None else ""
    return members, title


def epub_to_markdown(archive: zipfile.ZipFile) -> tuple[str, str]:
    """Convert the content documents of an EPUB container to Markdown and its title."""
    members, title = _epub_content_documents(archive)
    parts: list[str] = []
    for member in members:
        content = archive.read(member).decode("utf-8", errors="replace")
        markdown, _, _ = html_to_markdown(content)
        if markdown.strip():
            parts.append(markdown)
    return "\n\n".join(parts).strip() + "\n", title


def csv_to_markdown(doc_path: Path) -> str:
    """Convert a delimited text file to a Markdown table."""
    delimiter = "\t" if doc_path.suffix.lower() == ".tsv" else ","
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    with io.StringIO(text) as buffer:
        try:
            delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            logger.debug("Could not sniff the delimiter of %s, assuming %r.", doc_path, delimiter)
        rows = [
            row
            for row in csv.reader(buffer, delimiter=delimiter)
            if any(cell.strip() for cell in row)
        ]
    return _markdown_table(rows).strip() + "\n"


# ---------------------------------------------------------------------------
# Parsers.
# ---------------------------------------------------------------------------


class BaseParser:
    """Base class for the document parsers.

    A parser declares the file types it handles and the engine it belongs to, and converts one
    file to a `ParsedDocument`. `is_available` lets a parser whose dependency is missing decline
    the file so that the registry can fall back to the builtin engine.
    """

    ENGINE: ClassVar[str] = BUILTIN_ENGINE
    FILE_TYPES: ClassVar[frozenset[str]] = frozenset()
    UNAVAILABLE_HINT: ClassVar[str] = ""
    # Only the engines that call out to a configured processor need a `RAGLiteConfig`. Keeping it
    # optional means parsing a document never forces the (expensive) default config to be built.
    needs_config: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        """Return whether this parser's dependencies are installed."""
        return True

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert `doc_path` to a `ParsedDocument`."""
        error_message = f"{type(self).__name__} does not implement `parse`."
        raise NotImplementedError(error_message)


class MarkdownParser(BaseParser):
    """Read Markdown and plain text files as-is."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"md", "markdown", "mdown", "txt", "text", "log", "rst", "org"}
    )

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Read the file's content."""
        return ParsedDocument(
            content=doc_path.read_text(encoding="utf-8", errors="replace"),
            engine=self.ENGINE,
            file_type=file_type_of(doc_path),
        )


class PDFParser(BaseParser):
    """Parse a PDF with pdftext and reconstruct its headings and emphasis as Markdown."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset({"pdf"})

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Parse the PDF with pdftext and convert the parsed pages to Markdown."""
        from pdftext.extraction import dictionary_output

        from raglite._markdown import parsed_pdf_to_markdown

        pages = dictionary_output(doc_path, sort=True, keep_chars=False)
        return ParsedDocument(
            content="\n\n".join(parsed_pdf_to_markdown(pages)),
            engine=self.ENGINE,
            file_type=file_type_of(doc_path),
        )


class HtmlParser(BaseParser):
    """Convert HTML to Markdown with the standard library."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset({"html", "htm", "xhtml", "mhtml"})

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the HTML document to Markdown."""
        content, images, title = html_to_markdown(
            doc_path.read_text(encoding="utf-8", errors="replace")
        )
        return ParsedDocument(
            content=content,
            images=images,
            metadata={"title": title} if title else {},
            engine=self.ENGINE,
            file_type=file_type_of(doc_path),
        )


class CSVParser(BaseParser):
    """Convert delimited text files to Markdown tables."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset({"csv", "tsv"})

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the file to a Markdown table."""
        return ParsedDocument(
            content=csv_to_markdown(doc_path),
            engine=self.ENGINE,
            file_type=file_type_of(doc_path),
        )


class DocxParser(BaseParser):
    """Convert a .docx container to Markdown by reading its OOXML directly."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset({"docx"})

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the document's paragraphs and tables to Markdown."""
        with zipfile.ZipFile(doc_path) as archive:
            content = docx_to_markdown(archive)
            images = _container_images(archive, "word/media/")
        return ParsedDocument(
            content=content, images=images, engine=self.ENGINE, file_type=file_type_of(doc_path)
        )


class PPTXParser(BaseParser):
    """Convert a .pptx container to Markdown by reading its slides and notes."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset({"pptx"})

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the slides to Markdown."""
        with zipfile.ZipFile(doc_path) as archive:
            content = pptx_to_markdown(archive)
            images = _container_images(archive, "ppt/media/")
        return ParsedDocument(
            content=content, images=images, engine=self.ENGINE, file_type=file_type_of(doc_path)
        )


class XLSXParser(BaseParser):
    """Convert a .xlsx container to Markdown tables, one per sheet."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset({"xlsx", "xlsm"})

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the sheets to Markdown tables."""
        with zipfile.ZipFile(doc_path) as archive:
            content = xlsx_to_markdown(archive)
        return ParsedDocument(content=content, engine=self.ENGINE, file_type=file_type_of(doc_path))


class EPUBParser(BaseParser):
    """Convert an EPUB container to Markdown by walking its spine."""

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset({"epub"})

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the content documents of the EPUB to Markdown."""
        with zipfile.ZipFile(doc_path) as archive:
            content, title = epub_to_markdown(archive)
            images = _container_images(archive, "")
        return ParsedDocument(
            content=content,
            images=images,
            metadata={"title": title} if title else {},
            engine=self.ENGINE,
            file_type=file_type_of(doc_path),
        )


class ImageParser(BaseParser):
    """Reject images that no OCR or vision processor was configured for.

    The builtin engine has no image understanding, so this parser exists to produce the same
    actionable error a user needs: configure a multimodal or OCR document processor.
    """

    FILE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"}
    )
    UNAVAILABLE_HINT: ClassVar[str] = (
        "Configure a multimodal processor, e.g. "
        "`RAGLiteConfig(document_processor=MultimodalConfig(vlm='openai/gpt-4o-mini'))`."
    )

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Reject the image with an actionable error."""
        error_message = (
            f"To extract content from an image ({doc_path.name}), configure a multimodal or OCR "
            f"document processor, e.g. "
            "`RAGLiteConfig(document_processor=MultimodalConfig(vlm='openai/gpt-4o-mini'))`."
        )
        raise UnsupportedDocumentError(error_message)


class PandocParser(BaseParser):
    """Convert any format pandoc supports to GitHub Flavored Markdown."""

    ENGINE: ClassVar[str] = PANDOC_ENGINE
    FILE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "docx",
            "doc",
            "epub",
            "fb2",
            "html",
            "htm",
            "ipynb",
            "latex",
            "tex",
            "mediawiki",
            "odt",
            "opml",
            "org",
            "rst",
            "rtf",
            "textile",
            "wiki",
        }
    )
    UNAVAILABLE_HINT: ClassVar[str] = "Install the `pandoc` extra: `pip install 'raglite[pandoc]'`."

    @classmethod
    def is_available(cls) -> bool:
        """Return whether pypandoc and the pandoc binary are both available."""
        try:
            import pypandoc

            pypandoc.get_pandoc_version()
        except Exception:  # noqa: BLE001
            return False
        return True

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the document with pandoc."""
        import pypandoc

        try:
            content = pypandoc.convert_file(doc_path, to="gfm")
        except Exception as error:
            # Pandoc cannot read every file its format list suggests (a corrupt or mislabelled
            # file is the common case), so degrade to the builtin engine instead of failing.
            raise ParserFallbackError(str(error)) from error
        return ParsedDocument(content=content, engine=self.ENGINE, file_type=file_type_of(doc_path))


class MarkItDownParser(BaseParser):
    """Convert documents with Microsoft's MarkItDown library."""

    ENGINE: ClassVar[str] = MARKITDOWN_ENGINE
    FILE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"csv", "docx", "epub", "html", "htm", "json", "md", "pdf", "pptx", "xls", "xlsx", "xml"}
    )
    UNAVAILABLE_HINT: ClassVar[str] = "Install MarkItDown: `pip install markitdown`."

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the `markitdown` package is installed."""
        return importlib.util.find_spec("markitdown") is not None

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the document with MarkItDown."""
        from markitdown import MarkItDown

        result = MarkItDown().convert(str(doc_path))
        return ParsedDocument(
            content=result.markdown, engine=self.ENGINE, file_type=file_type_of(doc_path)
        )


class MultimodalParser(BaseParser):
    """Convert images and scanned PDFs with the OCR + VLM processor."""

    ENGINE: ClassVar[str] = MULTIMODAL_ENGINE
    FILE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif", "pdf"}
    )
    UNAVAILABLE_HINT: ClassVar[str] = (
        "Install the `multimodal` extra: `pip install 'raglite[multimodal]'`."
    )
    needs_config: ClassVar[bool] = True

    @classmethod
    def is_available(cls) -> bool:
        """Return whether an image rasterising backend is installed."""
        return any(
            importlib.util.find_spec(module) is not None for module in ("fitz", "pymupdf", "PIL")
        )

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the document with the multimodal processor."""
        from raglite._multimodal import is_multimodal_document, multimodal_to_markdown

        processor_config = getattr(config, "document_processor", None)
        if not isinstance(processor_config, MultimodalConfig):
            error_message = (
                "The multimodal parser requires "
                "`RAGLiteConfig(document_processor=MultimodalConfig(...))`."
            )
            raise UnsupportedDocumentError(error_message)
        if not is_multimodal_document(doc_path):
            error_message = f"The multimodal parser does not support file type: {doc_path}"
            raise UnsupportedDocumentError(error_message)
        return ParsedDocument(
            content=multimodal_to_markdown(doc_path, processor_config=processor_config),
            engine=self.ENGINE,
            file_type=file_type_of(doc_path),
        )


class MistralParser(BaseParser):
    """Convert documents with the Mistral OCR API."""

    ENGINE: ClassVar[str] = MISTRAL_ENGINE
    FILE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"pdf", "docx", "pptx", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"}
    )
    UNAVAILABLE_HINT: ClassVar[str] = "Install the `mistral-ocr` extra and set `MISTRAL_API_KEY`."

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the Mistral client is installed and an API key is configured."""
        if importlib.util.find_spec("mistralai") is None:
            return False
        return bool(os.environ.get("MISTRAL_API_KEY"))

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument:
        """Convert the document with Mistral OCR."""
        from raglite._mistral_ocr import SUPPORTED_EXTENSIONS, mistral_ocr_to_markdown

        processor_config = getattr(config, "document_processor", None)
        if not isinstance(processor_config, MistralOCRConfig):
            error_message = (
                "The Mistral parser requires "
                "`RAGLiteConfig(document_processor=MistralOCRConfig(...))`."
            )
            raise UnsupportedDocumentError(error_message)
        if doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            error_message = f"Mistral does not support file type: {doc_path}"
            raise UnsupportedDocumentError(error_message)
        return ParsedDocument(
            content=mistral_ocr_to_markdown(doc_path, processor_config=processor_config),
            engine=self.ENGINE,
            file_type=file_type_of(doc_path),
        )


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------

BUILTIN_PARSERS: tuple[type[BaseParser], ...] = (
    MarkdownParser,
    PDFParser,
    HtmlParser,
    CSVParser,
    DocxParser,
    PPTXParser,
    XLSXParser,
    EPUBParser,
    ImageParser,
)


class ParserEngineRegistry:
    """Map (engine, file type) pairs to parsers, falling back to the builtin engine.

    This is RAGLite's counterpart of WeKnora's `ParserEngineRegistry`: a requested engine that does
    not cover the file type, or whose dependency is missing, falls back to the builtin engine
    instead of failing; a file type no engine covers raises `UnsupportedDocumentError`.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._engines: dict[str, dict[str, type[BaseParser]]] = {}
        self._availability: dict[str, Callable[[], bool]] = {}
        self._hints: dict[str, str] = {}

    def register(
        self,
        engine: str,
        parsers: Iterator[type[BaseParser]] | Sequence[type[BaseParser]],
        *,
        check_available: Callable[[], bool] | None = None,
        unavailable_hint: str = "",
    ) -> None:
        """Register parsers under `engine`, keyed by the file types they declare."""
        mapping = self._engines.setdefault(engine, {})
        for parser in parsers:
            for file_type in parser.FILE_TYPES:
                mapping[file_type.lower()] = parser
        if check_available is not None:
            self._availability[engine] = check_available
        self._hints.setdefault(engine, unavailable_hint)

    def engines(self) -> tuple[str, ...]:
        """Return the registered engine names, builtin first."""
        return (
            BUILTIN_ENGINE,
            *sorted(engine for engine in self._engines if engine != BUILTIN_ENGINE),
        )

    def is_available(self, engine: str) -> bool:
        """Return whether `engine` is registered and its dependencies are installed."""
        if engine not in self._engines:
            return False
        check = self._availability.get(engine)
        return True if check is None else bool(check())

    def unavailable_hint(self, engine: str) -> str:
        """Return how to install the dependencies of `engine`."""
        return self._hints.get(engine, "")

    def file_types(self, *, engine: str | None = None) -> frozenset[str]:
        """Return the file types covered by `engine`, or by every engine when it is `None`."""
        if engine is not None:
            return frozenset(self._engines.get(engine, {}))
        return frozenset(file_type for mapping in self._engines.values() for file_type in mapping)

    def get_parser_class(self, engine: str | None, file_type: str) -> type[BaseParser]:
        """Return the parser class for (engine, file type), falling back to the builtin engine.

        WeKnora's registry has the same signature: it honours the requested engine when the engine
        covers the file type, and otherwise falls back to `builtin` before raising.
        """
        normalized = file_type.lower().lstrip(".")
        if engine and engine in self._engines and self.is_available(engine):
            parser_class = self._engines[engine].get(normalized)
            if parser_class is not None and parser_class.is_available():
                return parser_class
        builtin = self._engines.get(BUILTIN_ENGINE, {})
        parser_class = builtin.get(normalized)
        if parser_class is None:
            error_message = f"Unsupported file type: {file_type}"
            raise UnsupportedDocumentError(error_message)
        return parser_class

    def parser_for(self, doc_path: Path, *, engine: str | None = None) -> BaseParser:
        """Return a parser instance for `doc_path`."""
        parser_class = self.get_parser_class(engine, file_type_of(doc_path))
        return parser_class()

    def parse(
        self, doc_path: Path, *, engine: str | None = None, config: RAGLiteConfig | None = None
    ) -> ParsedDocument:
        """Parse `doc_path`, degrading to the builtin engine rather than failing.

        A requested engine that is unavailable, does not support the file type, or fails at
        runtime falls back to the builtin engine, which is precisely WeKnora's behaviour: the
        engine is an optimisation, never a correctness dependency.
        """
        requested = engine
        if requested and not self.is_available(requested):
            logger.debug(
                "Parser engine %r is unavailable (%s); falling back to the builtin engine for %s.",
                requested,
                self.unavailable_hint(requested),
                doc_path,
            )
            requested = None
        parser = self.parser_for(doc_path, engine=requested)
        try:
            return parser.parse(doc_path, config=config)
        except (UnsupportedDocumentError, ParserFallbackError) as error:
            if requested is None or requested == BUILTIN_ENGINE:
                raise
            # Only an engine's own failure is recoverable: the builtin engine is the floor.
            logger.warning(
                "Parser engine %r could not handle %s (%s); falling back to the builtin engine.",
                requested,
                doc_path,
                error,
            )
            fallback = self.parser_for(doc_path, engine=BUILTIN_ENGINE)
            return fallback.parse(doc_path, config=config)

    def report(self) -> dict[str, dict[str, Any]]:
        """Report every engine's availability, file types, and installation hint."""
        return {
            engine: {
                "available": self.is_available(engine),
                "file_types": sorted(self.file_types(engine=engine)),
                "hint": self.unavailable_hint(engine),
            }
            for engine in self.engines()
        }


def default_registry() -> ParserEngineRegistry:
    """Build the default parser registry: builtin parsers plus every installed engine."""
    registry = ParserEngineRegistry()
    registry.register(BUILTIN_ENGINE, BUILTIN_PARSERS)
    registry.register(
        PANDOC_ENGINE,
        [PandocParser],
        check_available=PandocParser.is_available,
        unavailable_hint=PandocParser.UNAVAILABLE_HINT,
    )
    registry.register(
        MARKITDOWN_ENGINE,
        [MarkItDownParser],
        check_available=MarkItDownParser.is_available,
        unavailable_hint=MarkItDownParser.UNAVAILABLE_HINT,
    )
    registry.register(
        MULTIMODAL_ENGINE,
        [MultimodalParser],
        check_available=MultimodalParser.is_available,
        unavailable_hint=MultimodalParser.UNAVAILABLE_HINT,
    )
    registry.register(
        MISTRAL_ENGINE,
        [MistralParser],
        check_available=MistralParser.is_available,
        unavailable_hint=MistralParser.UNAVAILABLE_HINT,
    )
    return registry


def parse_document(
    doc_path: Path, *, engine: str | None = None, config: RAGLiteConfig | None = None
) -> ParsedDocument:
    """Parse a document with the default parser registry.

    Parameters
    ----------
    doc_path
        Path to the document file.
    engine
        The parser engine to prefer. `None` uses the builtin engine.
    config
        Optional RAGLite configuration, needed by the multimodal and Mistral engines.

    Returns
    -------
    ParsedDocument
        The document's Markdown content and the images it carries.

    Raises
    ------
    UnsupportedDocumentError
        If no registered parser can handle the file.
    """
    return default_registry().parse(doc_path, engine=engine, config=config)


def supported_file_types(*, engine: str | None = None) -> frozenset[str]:
    """Return the file types the default registry can parse."""
    return default_registry().file_types(engine=engine)


def parser_report() -> dict[str, dict[str, Any]]:
    """Report the engines of the default registry and whether they are installed."""
    return default_registry().report()
