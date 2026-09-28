"""Multimodal (OCR + VLM) document processing for RAGLite.

This module turns images and scanned documents into first-class RAG sources by running two
complementary extraction stages over the same pixels:

1. **OCR** recovers the text that is *rendered inside* the image, so the pixels become
   searchable tokens.
2. **VLM** describes the *non-textual* content (charts, diagrams, screenshots, photographs),
   producing a natural-language annotation that can be embedded and retrieved.

Both stages are pluggable and local-first: the default OCR engine is an ONNX model that runs on
CPU without any cloud dependency, and the VLM slot accepts any vision-capable LiteLLM model,
cloud or locally served. Either stage can be disabled, and both engines can be injected directly
(see the `ocr_engine` and `vlm` fields of `MultimodalConfig`), which keeps the pipeline testable
and deterministic. A failing stage never discards the other stage's output: the two modalities
degrade independently.
"""

import base64
import importlib.util
import logging
import mimetypes
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from raglite._config import MultimodalConfig

logger = logging.getLogger(__name__)


# Extensions handled by this module: images are extracted directly, PDFs are rasterised first.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"})
PDF_EXTENSIONS = frozenset({".pdf"})
MULTIMODAL_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

# Default resolution used when rasterising PDF pages, expressed in dots per inch.
DEFAULT_PDF_DPI = 200

# Cap the number of worker threads used to extract multiple images in parallel.
MAX_EXTRACTION_WORKERS = 4

# OCR engine names accepted by `MultimodalConfig.ocr_engine`.
OCR_ENGINE_NAMES = frozenset({"auto", "rapidocr", "tesseract", "none"})


class MultimodalError(Exception):
    """Error during multimodal (OCR/VLM) document processing."""


@runtime_checkable
class OCREngine(Protocol):
    """Protocol for an optical character recognition engine."""

    def extract_text(self, image: bytes, *, mime_type: str = "image/png") -> str:
        """Extract the text rendered inside an encoded image.

        Parameters
        ----------
        image
            The encoded image bytes.
        mime_type
            The MIME type of the encoded image.

        Returns
        -------
        str
            The text found in the image, or an empty string when no text is found.
        """
        ...


@runtime_checkable
class VLMEngine(Protocol):
    """Protocol for a vision language model that describes an image in natural language."""

    def describe(self, image: bytes, *, mime_type: str, prompt: str) -> str:
        """Describe an encoded image in natural language.

        Parameters
        ----------
        image
            The encoded image bytes.
        mime_type
            The MIME type of the encoded image.
        prompt
            The instruction that steers the description.

        Returns
        -------
        str
            The image description.
        """
        ...


class ImageExtraction(BaseModel):
    """The multimodal extraction of a single image."""

    ocr_text: str = Field(default="", description="Text recovered from inside the image.")
    description: str = Field(default="", description="Natural-language description of the image.")
    image_type: str = Field(default="photo", description="Best-effort classification of the image.")

    def to_markdown(self) -> str:
        """Render the extraction as Markdown that keeps both modalities searchable.

        Returns
        -------
        str
            The extraction as Markdown, or an empty string when nothing was extracted.
        """
        blocks: list[str] = []
        if self.ocr_text.strip():
            blocks.append(self.ocr_text.strip())
        if self.description.strip():
            blocks.append(f"[Image ({self.image_type}): {self.description.strip()}]")
        return "\n\n".join(blocks)


def _module_is_available(module_name: str) -> bool:
    """Return whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):  # A parent package may itself be missing.
        return False


class RapidOCREngine:
    """Offline OCR engine backed by RapidOCR's ONNX models.

    RapidOCR bundles detection and recognition models that run on CPU through `onnxruntime`, so
    this engine needs neither cloud credentials nor system binaries. It covers Chinese and
    English out of the box, which makes it a good default for mixed-language corpora.
    """

    def __init__(self, *, languages: Sequence[str] = ("ch", "en")) -> None:
        """Initialise the engine.

        Parameters
        ----------
        languages
            The requested languages. RapidOCR's default model set already covers Chinese and
            English; the value is accepted for interface parity with the other engines.
        """
        self._engine: Any = None

    def _load(self) -> Any:
        """Load the RapidOCR engine lazily."""
        if self._engine is None:
            if not _module_is_available("rapidocr_onnxruntime"):
                error_msg = (
                    "To use the RapidOCR engine, please install the `multimodal` extra: "
                    "`pip install 'raglite[multimodal]'`."
                )
                raise ImportError(error_msg)
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    @staticmethod
    def _decode_image(image: bytes) -> Any:
        """Decode encoded image bytes into the array layout RapidOCR expects."""
        if not _module_is_available("cv2"):
            error_msg = (
                "To use the RapidOCR engine, please install the `multimodal` extra: "
                "`pip install 'raglite[multimodal]'`."
            )
            raise ImportError(error_msg)
        import cv2
        import numpy as np

        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            error_msg = "Could not decode the image bytes."
            raise MultimodalError(error_msg)
        return decoded

    @staticmethod
    def _collect_text(raw_output: Any) -> str:
        """Collect recognised text from the version-dependent RapidOCR output layout."""
        # RapidOCR 2.x returns an object exposing the recognised texts as `txts`.
        texts = getattr(raw_output, "txts", None)
        if texts is not None:
            return "\n".join(str(text) for text in texts if str(text).strip())
        # RapidOCR 1.x returns `(list[[box, text, score]] | None, elapsed)`.
        is_one_x_result = isinstance(raw_output, tuple) and len(raw_output) == 2  # noqa: PLR2004
        results = raw_output[0] if is_one_x_result else raw_output
        lines: list[str] = []
        for item in results or []:
            # Each 1.x entry is `[bounding_box, text, confidence_score]`.
            has_text_field = isinstance(item, (list, tuple)) and len(item) >= 2  # noqa: PLR2004
            text = item[1] if has_text_field else None
            if text is not None and str(text).strip():
                lines.append(str(text))
        return "\n".join(lines)

    def extract_text(self, image: bytes, *, mime_type: str = "image/png") -> str:
        """Extract the text rendered inside an encoded image with RapidOCR."""
        return self._collect_text(self._load()(self._decode_image(image)))


class TesseractOCREngine:
    """Offline OCR engine backed by a Tesseract installation through `pytesseract`."""

    def __init__(self, *, languages: Sequence[str] = ("eng",)) -> None:
        """Initialise the engine.

        Parameters
        ----------
        languages
            The Tesseract language packs to enable, e.g. `("eng", "chi_sim")`.
        """
        self._languages = "+".join(languages)

    def extract_text(self, image: bytes, *, mime_type: str = "image/png") -> str:
        """Extract the text rendered inside an encoded image with Tesseract."""
        if not (_module_is_available("pytesseract") and _module_is_available("PIL")):
            error_msg = "To use the Tesseract engine, please install `pytesseract` and `pillow`."
            raise ImportError(error_msg)
        import pytesseract
        from PIL import Image

        with Image.open(BytesIO(image)) as pil_image:
            return str(pytesseract.image_to_string(pil_image, lang=self._languages))


class NoOCREngine:
    """OCR engine that extracts nothing, used to index images by VLM description alone."""

    def extract_text(self, image: bytes, *, mime_type: str = "image/png") -> str:
        """Return an empty string without reading the image."""
        return ""


class LiteLLMVLMEngine:
    """Vision language model engine backed by any vision-capable LiteLLM model.

    The engine sends the image as a base64 data URI together with the prompt, so it works with
    hosted providers (`openai/gpt-4o-mini`, `anthropic/claude-...`, `gemini/...`) as well as
    locally served OpenAI-compatible models (`ollama/llava`, or `openai/<model>` with an
    `api_base` that points at a local server).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        num_retries: int = 3,
    ) -> None:
        """Initialise the engine.

        Parameters
        ----------
        model
            The LiteLLM model identifier, e.g. `openai/gpt-4o-mini`.
        api_key
            The API key. Falls back to the provider-specific environment variable when omitted.
        api_base
            The base URL of an OpenAI-compatible endpoint.
        max_tokens
            The maximum number of tokens in the description.
        temperature
            The sampling temperature.
        num_retries
            The number of retries per image before giving up.
        """
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._num_retries = num_retries

    def describe(self, image: bytes, *, mime_type: str, prompt: str) -> str:
        """Describe an encoded image with a vision-capable LiteLLM model."""
        import litellm

        encoded = base64.standard_b64encode(image).decode("ascii")
        messages: list[Any] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            }
        ]
        response = litellm.completion(
            model=self._model,
            messages=messages,
            api_key=self._api_key,
            api_base=self._api_base,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            num_retries=self._num_retries,
        )
        content: Any = response.choices[0].message.content  # type: ignore[union-attr]
        if isinstance(content, list):  # Some providers return content parts instead of a string.
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return str(content or "").strip()


class StaticVLMEngine:
    """VLM engine that returns a fixed description, used for tests and dry runs."""

    def __init__(self, description: str) -> None:
        """Initialise the engine.

        Parameters
        ----------
        description
            The description returned for every image.
        """
        self._description = description

    def describe(self, image: bytes, *, mime_type: str, prompt: str) -> str:
        """Return the configured description."""
        return self._description


def resolve_ocr_engine(processor_config: MultimodalConfig) -> OCREngine:
    """Resolve the configured OCR engine, instantiating it when a name is given.

    Parameters
    ----------
    processor_config
        The multimodal processor configuration.

    Returns
    -------
    OCREngine
        The resolved OCR engine.

    Raises
    ------
    MultimodalError
        If the engine name is unknown, or if `auto` finds no installed engine.
    """
    engine = processor_config.ocr_engine
    if engine is not None and not isinstance(engine, str):
        return engine  # An injected engine takes precedence over everything else.
    name = ("auto" if engine is None else engine).lower()
    if name not in OCR_ENGINE_NAMES:
        error_msg = f"Unknown OCR engine: {name}. Expected one of: {sorted(OCR_ENGINE_NAMES)}."
        raise MultimodalError(error_msg)
    if name == "none":
        return NoOCREngine()
    if name in ("auto", "rapidocr") and _module_is_available("rapidocr_onnxruntime"):
        return RapidOCREngine(languages=processor_config.ocr_languages)
    if name in ("auto", "tesseract") and _module_is_available("pytesseract"):
        return TesseractOCREngine(languages=processor_config.tesseract_languages)
    if name == "rapidocr":
        error_msg = (
            "The requested RapidOCR engine is not installed. Please install the `multimodal` "
            "extra: `pip install 'raglite[multimodal]'`."
        )
        raise MultimodalError(error_msg)
    if name == "tesseract":
        error_msg = "The requested Tesseract engine is not installed: `pip install pytesseract`."
        raise MultimodalError(error_msg)
    error_msg = (
        "No OCR engine is installed. Please install the `multimodal` extra: "
        "`pip install 'raglite[multimodal]'`, or set `ocr_engine='none'` to index images by "
        "description alone."
    )
    raise MultimodalError(error_msg)


def resolve_vlm_engine(processor_config: MultimodalConfig) -> VLMEngine | None:
    """Resolve the configured VLM engine, returning None when descriptions are disabled.

    Parameters
    ----------
    processor_config
        The multimodal processor configuration.

    Returns
    -------
    VLMEngine | None
        The resolved VLM engine, or None when no description should be produced.
    """
    if not processor_config.describe_images or processor_config.vlm is None:
        return None
    if isinstance(processor_config.vlm, str):
        return LiteLLMVLMEngine(
            model=processor_config.vlm,
            api_key=processor_config.vlm_api_key,
            api_base=processor_config.vlm_api_base,
            max_tokens=processor_config.vlm_max_tokens,
            temperature=processor_config.vlm_temperature,
            num_retries=processor_config.vlm_num_retries,
        )
    return processor_config.vlm


def classify_image(filename: str) -> str:
    """Classify an image into a coarse type that is used as a retrieval hint.

    Parameters
    ----------
    filename
        The image filename, which often encodes its role in a document.

    Returns
    -------
    str
        One of `chart`, `diagram`, `table`, `screenshot`, `logo`, or `photo`.
    """
    stem = filename.lower()
    for image_type, keywords in (
        ("chart", ("chart", "graph", "plot", "figure", "fig")),
        ("diagram", ("diagram", "flow", "schematic", "arch")),
        ("table", ("table", "sheet", "grid")),
        ("screenshot", ("screen", "shot", "capture")),
        ("logo", ("logo", "icon")),
    ):
        if any(keyword in stem for keyword in keywords):
            return image_type
    return "photo"


def extract_image(
    image: bytes,
    *,
    processor_config: MultimodalConfig,
    mime_type: str = "image/png",
    filename: str = "",
) -> ImageExtraction:
    """Run the OCR and VLM stages over a single encoded image.

    Parameters
    ----------
    image
        The encoded image bytes.
    processor_config
        The multimodal processor configuration.
    mime_type
        The MIME type of the encoded image.
    filename
        The original filename, used to classify the image type.

    Returns
    -------
    ImageExtraction
        The recovered text, the visual description, and the image type.
    """
    ocr_text = ""
    if processor_config.include_ocr_text:
        try:
            ocr_text = resolve_ocr_engine(processor_config).extract_text(image, mime_type=mime_type)
        # Any engine failure (missing model, undecodable image, inference error) must degrade to
        # the other modality rather than abort the whole document.
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR failed (%s); continuing with the image description.", e)

    description = ""
    if (vlm_engine := resolve_vlm_engine(processor_config)) is not None:
        try:
            description = vlm_engine.describe(
                image, mime_type=mime_type, prompt=processor_config.vlm_prompt
            )
        # Any provider failure (network, quota, unsupported model) must degrade to the other
        # modality rather than abort the whole document.
        except Exception as e:  # noqa: BLE001
            logger.warning("Image description failed (%s); continuing with the OCR text.", e)

    return ImageExtraction(
        ocr_text=ocr_text,
        description=description,
        image_type=classify_image(filename),
    )


def extract_images(
    images: Sequence[bytes],
    *,
    processor_config: MultimodalConfig,
    mime_type: str = "image/png",
    filenames: Sequence[str] | None = None,
) -> list[ImageExtraction]:
    """Run the OCR and VLM stages over a sequence of images.

    Images are extracted concurrently because the VLM stage is dominated by network latency, so
    describing the pages of a scanned document sequentially wastes most of the wall-clock time.

    Parameters
    ----------
    images
        The encoded image bytes.
    processor_config
        The multimodal processor configuration.
    mime_type
        The MIME type of the encoded images.
    filenames
        The original filenames, used to classify the image types.

    Returns
    -------
    list[ImageExtraction]
        The extractions, in the same order as the inputs.
    """
    names = list(filenames) if filenames is not None else [""] * len(images)
    max_workers = min(processor_config.extraction_max_workers, len(images))
    if max_workers <= 1:
        return [
            extract_image(
                image, processor_config=processor_config, mime_type=mime_type, filename=name
            )
            for image, name in zip(images, names, strict=True)
        ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                extract_image,
                image,
                processor_config=processor_config,
                mime_type=mime_type,
                filename=name,
            )
            for image, name in zip(images, names, strict=True)
        ]
        return [future.result() for future in futures]


@dataclass(frozen=True)
class RasterizedPage:
    """A PDF page rasterised into an image."""

    image: bytes
    mime_type: str
    page_number: int


def rasterize_pdf(pdf_path: Path, *, dpi: int = DEFAULT_PDF_DPI) -> list[RasterizedPage]:
    """Rasterise the pages of a PDF into PNG images.

    Parameters
    ----------
    pdf_path
        The path to the PDF file.
    dpi
        The rasterisation resolution in dots per inch.

    Returns
    -------
    list[RasterizedPage]
        One rasterised image per page, in document order.

    Raises
    ------
    ImportError
        If no supported PDF rasteriser is installed.
    """
    if _module_is_available("pymupdf"):
        import pymupdf
    elif _module_is_available("fitz"):
        import fitz as pymupdf  # type: ignore[no-redef]
    else:
        error_msg = (
            "To process PDFs with the multimodal processor, please install the `multimodal` "
            "extra: `pip install 'raglite[multimodal]'`."
        )
        raise ImportError(error_msg)
    # PyMuPDF ships no type information for its `Document` factory, so treat it as dynamic.
    pymupdf_module: Any = pymupdf
    with pymupdf_module.open(pdf_path) as pdf:
        return [
            RasterizedPage(
                image=page.get_pixmap(dpi=dpi).tobytes("png"),
                mime_type="image/png",
                page_number=page_number,
            )
            for page_number, page in enumerate(pdf.pages(), start=1)
        ]


def _guess_mime_type(path: Path) -> str:
    """Guess the MIME type of a path, defaulting to PNG for unknown image extensions."""
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/png"


def image_to_markdown(image_path: Path, *, processor_config: MultimodalConfig) -> str:
    """Convert a single image to Markdown with OCR text and a VLM description.

    Parameters
    ----------
    image_path
        The path to the image file.
    processor_config
        The multimodal processor configuration.

    Returns
    -------
    str
        The extracted content as Markdown.
    """
    return extract_image(
        image_path.read_bytes(),
        processor_config=processor_config,
        mime_type=_guess_mime_type(image_path),
        filename=image_path.name,
    ).to_markdown()


def multimodal_to_markdown(doc_path: Path, *, processor_config: MultimodalConfig) -> str:
    """Convert an image or PDF to Markdown with the multimodal (OCR + VLM) pipeline.

    Parameters
    ----------
    doc_path
        The path to the image or PDF file.
    processor_config
        The multimodal processor configuration.

    Returns
    -------
    str
        The extracted content as Markdown.

    Raises
    ------
    MultimodalError
        If the file extension is not handled by this module.
    """
    suffix = doc_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return image_to_markdown(doc_path, processor_config=processor_config)
    if suffix in PDF_EXTENSIONS:
        pages = rasterize_pdf(doc_path, dpi=processor_config.pdf_dpi)
        if processor_config.max_pdf_pages is not None:
            pages = pages[: processor_config.max_pdf_pages]
        extractions = extract_images(
            [page.image for page in pages],
            processor_config=processor_config,
            filenames=[f"{doc_path.stem}-page-{page.page_number}" for page in pages],
        )
        markdown_pages = [extraction.to_markdown() for extraction in extractions]
        return "\n\n".join(page for page in markdown_pages if page.strip())
    error_msg = f"The multimodal processor does not support the file type: {doc_path.name}"
    raise MultimodalError(error_msg)


def is_multimodal_document(doc_path: Path) -> bool:
    """Return whether a path is handled by the multimodal processor.

    Parameters
    ----------
    doc_path
        The path to the document file.

    Returns
    -------
    bool
        Whether the file is an image or a PDF.
    """
    return doc_path.suffix.lower() in MULTIMODAL_EXTENSIONS
