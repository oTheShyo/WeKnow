"""Convert any document to Markdown."""

import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from raglite._config import MistralOCRConfig, MultimodalConfig, RAGLiteConfig
from raglite._parsers import MISTRAL_ENGINE, MULTIMODAL_ENGINE, parse_document

logger = logging.getLogger(__name__)


def parsed_pdf_to_markdown(pages: list[dict[str, Any]]) -> list[str]:  # noqa: C901, PLR0915
    """Convert a PDF parsed with pdftext to Markdown."""

    def add_heading_level_metadata(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: C901
        """Add heading level metadata to a PDF parsed with pdftext."""

        def extract_font_size(span: dict[str, Any]) -> float:
            """Extract the font size from a text span."""
            font_size: float = 1.0
            if span["font"]["size"] > 1:  # A value of 1 appears to mean "unknown" in pdftext.
                font_size = span["font"]["size"]
            elif digit_sequences := re.findall(r"\d+", span["font"]["name"] or ""):
                font_size = float(digit_sequences[-1])
            elif "\n" not in span["text"]:  # Occasionally a span can contain a newline character.
                if round(span["rotation"]) in (0.0, 180.0, -180.0):
                    font_size = span["bbox"][3] - span["bbox"][1]
                elif round(span["rotation"]) in (90.0, -90.0, 270.0, -270.0):
                    font_size = span["bbox"][2] - span["bbox"][0]
            return font_size

        # Copy the pages.
        pages = deepcopy(pages)
        # Extract an array of all font sizes used by the text spans.
        font_sizes = np.asarray(
            [
                extract_font_size(span)
                for page in pages
                for block in page["blocks"]
                for line in block["lines"]
                for span in line["spans"]
            ]
        )
        font_sizes = np.round(font_sizes * 2) / 2
        unique_font_sizes, counts = np.unique(font_sizes, return_counts=True)
        # Determine the paragraph font size as the mode font size.
        tiny = unique_font_sizes < min(5, np.max(unique_font_sizes))
        counts[tiny] = -counts[tiny]
        mode = np.argmax(counts)
        counts[tiny] = -counts[tiny]
        mode_font_size = unique_font_sizes[mode]
        # Determine (at most) 6 heading font sizes by clustering font sizes larger than the mode.
        heading_font_sizes = unique_font_sizes[mode + 1 :]
        if len(heading_font_sizes) > 0:
            heading_counts = counts[mode + 1 :]
            kmeans = KMeans(n_clusters=min(6, len(heading_font_sizes)), random_state=42)
            kmeans.fit(heading_font_sizes[:, np.newaxis], sample_weight=heading_counts)
            heading_font_sizes = np.sort(np.ravel(kmeans.cluster_centers_))[::-1]
        # Add heading level information to the text spans and lines.
        for page in pages:
            for block in page["blocks"]:
                for line in block["lines"]:
                    if "md" not in line:
                        line["md"] = {}
                    heading_level = np.zeros(8)  # 0-5: <h1>-<h6>, 6: <p>, 7: <small>
                    for span in line["spans"]:
                        if "md" not in span:
                            span["md"] = {}
                        span_font_size = extract_font_size(span)
                        if span_font_size < mode_font_size:
                            idx = 7
                        elif span_font_size == mode_font_size or len(heading_font_sizes) == 0:
                            idx = 6
                        else:
                            idx = np.argmin(np.abs(heading_font_sizes - span_font_size))  # type: ignore[assignment]
                        span["md"]["heading_level"] = idx + 1
                        heading_level[idx] += len(span["text"])
                    line["md"]["heading_level"] = np.argmax(heading_level) + 1
        return pages

    def add_emphasis_metadata(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add emphasis metadata such as bold and italic to a PDF parsed with pdftext."""
        # Copy the pages.
        pages = deepcopy(pages)
        # Add emphasis metadata to the text spans.
        for page in pages:
            for block in page["blocks"]:
                for line in block["lines"]:
                    if "md" not in line:
                        line["md"] = {}
                    for span in line["spans"]:
                        if "md" not in span:
                            span["md"] = {}
                        span["md"]["bold"] = span["font"]["weight"] > 500  # noqa: PLR2004
                        span["md"]["italic"] = "ital" in (span["font"]["name"] or "").lower()
                    line["md"]["bold"] = all(
                        span["md"]["bold"] for span in line["spans"] if span["text"].strip()
                    )
                    line["md"]["italic"] = all(
                        span["md"]["italic"] for span in line["spans"] if span["text"].strip()
                    )
        return pages

    def strip_page_numbers(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip page numbers from a PDF parsed with pdftext."""
        # Copy the pages.
        pages = deepcopy(pages)
        # Remove lines that only contain a page number.
        for page in pages:
            for block in page["blocks"]:
                block["lines"] = [
                    line
                    for line in block["lines"]
                    if not re.match(
                        r"^\s*[#0]*\d+\s*$", "".join(span["text"] for span in line["spans"])
                    )
                ]
        return pages

    def convert_to_markdown(pages: list[dict[str, Any]]) -> list[str]:  # noqa: C901, PLR0912
        """Convert a list of pages to Markdown."""
        pages_md = []
        for page in pages:
            page_md = ""
            for block in page["blocks"]:
                block_text = ""
                for line in block["lines"]:
                    # Build the line text and style the spans.
                    line_text = ""
                    for span in line["spans"]:
                        if (
                            not line["md"]["bold"]
                            and not line["md"]["italic"]
                            and span["md"]["bold"]
                            and span["md"]["italic"]
                        ):
                            line_text += f"***{span['text']}***"
                        elif not line["md"]["bold"] and span["md"]["bold"]:
                            line_text += f"**{span['text']}**"
                        elif not line["md"]["italic"] and span["md"]["italic"]:
                            line_text += f"*{span['text']}*"
                        else:
                            line_text += span["text"]
                    # Add emphasis to the line (if it's not a heading or whitespace).
                    line_text = line_text.rstrip()
                    line_is_whitespace = not line_text.strip()
                    line_is_heading = line["md"]["heading_level"] <= 6  # noqa: PLR2004
                    if not line_is_heading and not line_is_whitespace:
                        if line["md"]["bold"] and line["md"]["italic"]:
                            line_text = f"***{line_text}***"
                        elif line["md"]["bold"]:
                            line_text = f"**{line_text}**"
                        elif line["md"]["italic"]:
                            line_text = f"*{line_text}*"
                    # Set the heading level.
                    if line_is_heading and not line_is_whitespace:
                        line_text = f"{'#' * line['md']['heading_level']} {line_text}"
                    line_text += "\n"
                    block_text += line_text
                block_text = block_text.rstrip() + "\n\n"
                page_md += block_text
            pages_md.append(page_md.strip())
        return pages_md

    def merge_split_headings(pages: list[str]) -> list[str]:
        """Merge headings that are split across lines."""

        def _merge_split_headings(match: re.Match[str]) -> str:
            atx_headings = [line.strip("# ").strip() for line in match.group().splitlines()]
            return f"{match.group(1)} {' '.join(atx_headings)}\n\n"

        pages_md = [
            re.sub(
                r"^(#+)[ \t]+[^\n]+\n+(?:^\1[ \t]+[^\n]+\n+)+",
                _merge_split_headings,
                page,
                flags=re.MULTILINE,
            )
            for page in pages
        ]
        return pages_md

    # Add heading level metadata.
    pages = add_heading_level_metadata(pages)
    # Add emphasis metadata.
    pages = add_emphasis_metadata(pages)
    # Strip page numbers.
    pages = strip_page_numbers(pages)
    # Convert the pages to Markdown.
    pages_md = convert_to_markdown(pages)
    # Merge headings that are split across lines.
    pages_md = merge_split_headings(pages_md)
    return pages_md


def _preferred_parser_engine(doc_path: Path, config: RAGLiteConfig) -> str | None:
    """Return the parser engine implied by the config and the configured document processor.

    `None` means "let the registry choose", which resolves to the builtin engine. The multimodal
    and Mistral engines only cover the file types they actually support: for anything else the
    registry falls back to the builtin engine, so a PDF still parses with pdftext when a
    multimodal processor is configured but the file is not a scan.
    """
    from raglite._mistral_ocr import SUPPORTED_EXTENSIONS
    from raglite._multimodal import is_multimodal_document

    if config.parser_engine:
        return config.parser_engine
    if isinstance(config.document_processor, MultimodalConfig) and is_multimodal_document(doc_path):
        return MULTIMODAL_ENGINE
    if isinstance(config.document_processor, MistralOCRConfig) and (
        doc_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        return MISTRAL_ENGINE
    return None


def document_to_markdown(doc_path: Path, *, config: RAGLiteConfig | None = None) -> str:
    """Convert any document to GitHub Flavored Markdown.

    Parameters
    ----------
    doc_path
        Path to the document file.
    config
        Optional RAGLite configuration. If document_processor is set to a
        MultimodalConfig, runs the OCR + VLM pipeline over images and PDFs. If it is set to a
        MistralOCRConfig, uses MistralOCR instead of the default processor. `parser_engine`
        overrides the parser engine for every format. Parsing does not need a config: only the
        multimodal and Mistral engines do.

    Returns
    -------
    str
        Document content as GitHub Flavored Markdown.

    Raises
    ------
    ValueError
        If the document is an image and no multimodal document processor is configured.
    """
    engine = None if config is None else _preferred_parser_engine(doc_path, config)
    if engine is None and config is not None and config.document_processor is not None:
        logger.debug(
            "The configured document processor does not support file type: %s\n"
            "Falling back to the default processor.",
            doc_path,
        )
    # The registry picks the parser, verifies that its engine is installed, and falls back to the
    # builtin engine otherwise. The builtin image parser is what guards against silently decoding
    # binary image data as text: it raises instead of returning garbage.
    return parse_document(doc_path, engine=engine, config=config).content
