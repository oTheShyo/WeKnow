"""Test multimodal (OCR + VLM) document processing."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from weknow import (
    Document,
    ImageExtraction,
    LiteLLMVLMEngine,
    MultimodalConfig,
    MultimodalError,
    RAGLiteConfig,
    StaticVLMEngine,
    classify_image,
    extract_image,
    extract_images,
    is_multimodal_document,
    multimodal_to_markdown,
    rasterize_pdf,
)
from weknow._config import DEFAULT_VLM_PROMPT
from weknow._markdown import document_to_markdown
from weknow._multimodal import NoOCREngine, resolve_ocr_engine, resolve_vlm_engine

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

NO_OCR_ENGINE_INSTALLED = (
    importlib.util.find_spec("rapidocr_onnxruntime") is None
    and importlib.util.find_spec("pytesseract") is None
)
NO_PDF_RASTERIZER_INSTALLED = importlib.util.find_spec("pymupdf") is None


class StubOCREngine:
    """OCR engine that returns a fixed text, optionally failing."""

    def __init__(self, text: str = "", *, fails: bool = False) -> None:
        self.text = text
        self.fails = fails
        self.calls: list[str] = []

    def extract_text(self, image: bytes, *, mime_type: str = "image/png") -> str:
        """Return the configured text."""
        self.calls.append(mime_type)
        if self.fails:
            error_msg = "stub OCR failure"
            raise MultimodalError(error_msg)
        return self.text


class StubVLMEngine:
    """VLM engine that echoes the first byte of the image, optionally failing."""

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.calls: list[str] = []

    def describe(self, image: bytes, *, mime_type: str, prompt: str) -> str:
        """Return a description derived from the image bytes."""
        self.calls.append(prompt)
        if self.fails:
            error_msg = "stub VLM failure"
            raise MultimodalError(error_msg)
        return f"desc-{image[:1].decode()}"


def _config(**kwargs: Any) -> RAGLiteConfig:
    """Build a RAGLite config that never downloads or loads a model."""
    return RAGLiteConfig(
        llm="gpt-4o-mini", embedder="text-embedding-3-small", reranker=None, **kwargs
    )


def _processor(**kwargs: Any) -> MultimodalConfig:
    """Build a multimodal config with injected stub engines by default."""
    defaults: dict[str, Any] = {
        "ocr_engine": StubOCREngine("invoice total 42"),
        "vlm": StaticVLMEngine("A table with two columns"),
    }
    return MultimodalConfig(**{**defaults, **kwargs})


def test_extract_image_merges_both_modalities() -> None:
    """Both the OCR text and the description end up in the Markdown."""
    extraction = extract_image(b"fake", processor_config=_processor(), filename="chart-1.png")
    markdown = extraction.to_markdown()
    assert extraction.ocr_text == "invoice total 42"
    assert extraction.description == "A table with two columns"
    assert extraction.image_type == "chart"
    assert markdown.startswith("invoice total 42")
    assert "[Image (chart): A table with two columns]" in markdown


def test_extract_image_ocr_only() -> None:
    """Without a VLM the extraction keeps the OCR text and no description block."""
    markdown = extract_image(b"fake", processor_config=_processor(vlm=None)).to_markdown()
    assert markdown == "invoice total 42"


def test_extract_image_description_only() -> None:
    """With OCR disabled the extraction is a pure VLM description."""
    extraction = extract_image(b"fake", processor_config=_processor(ocr_engine="none"))
    assert extraction.ocr_text == ""
    assert extraction.to_markdown() == "[Image (photo): A table with two columns]"


def test_extract_image_with_nothing_extracted() -> None:
    """An empty extraction renders as an empty string."""
    config = _processor(ocr_engine="none", vlm=None)
    assert extract_image(b"fake", processor_config=config).to_markdown() == ""
    assert ImageExtraction().to_markdown() == ""


def test_extract_image_survives_ocr_failure(caplog: pytest.LogCaptureFixture) -> None:
    """A failing OCR stage degrades to the description instead of raising."""
    config = _processor(ocr_engine=StubOCREngine(fails=True))
    with caplog.at_level("WARNING"):
        extraction = extract_image(b"fake", processor_config=config)
    assert extraction.ocr_text == ""
    assert extraction.description == "A table with two columns"
    assert "OCR failed" in caplog.text


def test_extract_image_survives_vlm_failure(caplog: pytest.LogCaptureFixture) -> None:
    """A failing VLM stage degrades to the OCR text instead of raising."""
    config = _processor(vlm=StubVLMEngine(fails=True))
    with caplog.at_level("WARNING"):
        extraction = extract_image(b"fake", processor_config=config)
    assert extraction.ocr_text == "invoice total 42"
    assert extraction.description == ""
    assert "Image description failed" in caplog.text


def test_extract_images_preserves_order() -> None:
    """Parallel extraction returns the extractions in input order."""
    images = [b"aaaa", b"bbbb", b"cccc", b"dddd"]
    config = _processor(vlm=StubVLMEngine(), extraction_max_workers=4)
    extractions = extract_images(images, processor_config=config)
    assert [extraction.description for extraction in extractions] == [
        "desc-a",
        "desc-b",
        "desc-c",
        "desc-d",
    ]


def test_extract_images_single_image_path() -> None:
    """A single image runs on the calling thread, not through the executor."""
    extractions = extract_images([b"zzzz"], processor_config=_processor(vlm=StubVLMEngine()))
    assert [extraction.description for extraction in extractions] == ["desc-z"]


def test_classify_image() -> None:
    """Image filenames map onto coarse retrieval hints."""
    assert classify_image("quarterly-revenue-chart.png") == "chart"
    assert classify_image("system-architecture-diagram.png") == "diagram"
    assert classify_image("table-3.png") == "table"
    assert classify_image("screen-capture.png") == "screenshot"
    assert classify_image("acme-logo.png") == "logo"
    assert classify_image("photo-2024.png") == "photo"


def test_is_multimodal_document() -> None:
    """Only images and PDFs are handled by the multimodal processor."""
    assert is_multimodal_document(Path("scan.PNG"))
    assert is_multimodal_document(Path("report.pdf"))
    assert not is_multimodal_document(Path("notes.md"))
    assert not is_multimodal_document(Path("deck.pptx"))


def test_resolve_ocr_engine_none_and_injected() -> None:
    """The `none` engine extracts nothing and an injected engine is returned as-is."""
    assert isinstance(resolve_ocr_engine(MultimodalConfig(ocr_engine="none")), NoOCREngine)
    stub = StubOCREngine("x")
    assert resolve_ocr_engine(MultimodalConfig(ocr_engine=stub)) is stub


def test_resolve_ocr_engine_rejects_unknown_name() -> None:
    """An unknown OCR engine name is reported as such."""
    with pytest.raises(MultimodalError, match="Unknown OCR engine"):
        resolve_ocr_engine(MultimodalConfig(ocr_engine="magic-ocr"))


@pytest.mark.skipif(not NO_OCR_ENGINE_INSTALLED, reason="an OCR engine is installed")
def test_resolve_ocr_engine_without_installation() -> None:
    """`auto` fails loudly when no OCR engine is installed."""
    with pytest.raises(MultimodalError, match="No OCR engine is installed"):
        resolve_ocr_engine(MultimodalConfig(ocr_engine="auto"))


def test_resolve_vlm_engine() -> None:
    """The VLM engine honours the model string, the kill switch, and injection."""
    assert (
        resolve_vlm_engine(MultimodalConfig(describe_images=False, vlm="openai/gpt-4o-mini"))
        is None
    )
    assert resolve_vlm_engine(MultimodalConfig(vlm=None)) is None
    engine = resolve_vlm_engine(
        MultimodalConfig(vlm="openai/gpt-4o-mini", vlm_api_key="k", vlm_api_base="http://localhost")
    )
    assert isinstance(engine, LiteLLMVLMEngine)
    stub = StaticVLMEngine("d")
    assert resolve_vlm_engine(MultimodalConfig(vlm=stub)) is stub


def test_multimodal_to_markdown_image_file(tmp_path: Path) -> None:
    """An image file is converted to Markdown with both modalities."""
    image_path = tmp_path / "diagram-flow.png"
    image_path.write_bytes(b"fake-image")
    markdown = multimodal_to_markdown(image_path, processor_config=_processor())
    assert "invoice total 42" in markdown
    assert "[Image (diagram): A table with two columns]" in markdown


def test_multimodal_to_markdown_rejects_unsupported_extension(tmp_path: Path) -> None:
    """An unsupported extension is reported instead of being silently read as text."""
    doc_path = tmp_path / "notes.docx"
    doc_path.write_bytes(b"fake-docx")
    with pytest.raises(MultimodalError, match="does not support the file type"):
        multimodal_to_markdown(doc_path, processor_config=_processor())


def test_document_to_markdown_routes_image_to_multimodal(tmp_path: Path) -> None:
    """The configured multimodal processor is used for image documents."""
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"fake-image")
    markdown = document_to_markdown(image_path, config=_config(document_processor=_processor()))
    assert "invoice total 42" in markdown


def test_document_to_markdown_rejects_image_without_processor(tmp_path: Path) -> None:
    """An image without a multimodal processor raises an actionable error."""
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"fake-image")
    with pytest.raises(ValueError, match="multimodal or OCR document processor"):
        document_to_markdown(image_path, config=_config())


def test_document_from_path_indexes_image_content(tmp_path: Path) -> None:
    """An image becomes a searchable document whose content carries both modalities."""
    image_path = tmp_path / "revenue-chart.png"
    image_path.write_bytes(b"fake-image")
    document = Document.from_path(
        image_path,
        config=_config(document_processor=_processor()),
    )
    assert document.content is not None
    assert "invoice total 42" in document.content
    assert "A table with two columns" in document.content
    assert document.metadata_["filename"] == ["revenue-chart.png"]


@pytest.mark.skipif(NO_PDF_RASTERIZER_INSTALLED, reason="pymupdf is not installed")
def test_rasterize_pdf() -> None:
    """PDF pages are rasterised into PNG images in document order."""
    pages = rasterize_pdf(Path(__file__).parent / "specrel.pdf", dpi=50)
    assert len(pages) >= 1
    assert [page.page_number for page in pages] == list(range(1, len(pages) + 1))
    assert all(page.mime_type == "image/png" for page in pages)
    assert all(page.image.startswith(PNG_MAGIC) for page in pages)


@pytest.mark.skipif(NO_PDF_RASTERIZER_INSTALLED, reason="pymupdf is not installed")
def test_multimodal_to_markdown_pdf() -> None:
    """A PDF is processed page by page and honours the page cap."""
    pdf_path = Path(__file__).parent / "specrel.pdf"
    processor_config = _processor(pdf_dpi=50, max_pdf_pages=1, extraction_max_workers=2)
    markdown = multimodal_to_markdown(pdf_path, processor_config=processor_config)
    assert markdown.count("[Image (") == 1  # Exactly one page was extracted.
    assert "invoice total 42" in markdown


def test_default_vlm_prompt_is_used() -> None:
    """The configured prompt is what reaches the vision model."""
    stub = StubVLMEngine()
    extract_image(b"fake", processor_config=_processor(vlm=stub))
    assert stub.calls == [DEFAULT_VLM_PROMPT]
