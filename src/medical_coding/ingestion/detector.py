"""Document format detection via file extensions, MIME signatures, and magic byte analysis."""

from enum import StrEnum
from pathlib import Path


class DocumentFormat(StrEnum):
    """Supported input document formats for clinical discharge summaries."""

    PDF = "PDF"
    TXT = "TXT"
    IMAGE = "IMAGE"
    MANUAL_TEXT = "MANUAL_TEXT"
    UNKNOWN = "UNKNOWN"


# Magic byte signatures for binary formats
_MAGIC_SIGNATURES: list[tuple[bytes, DocumentFormat, str]] = [
    (b"%PDF", DocumentFormat.PDF, "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", DocumentFormat.IMAGE, "image/png"),
    (b"\xff\xd8\xff", DocumentFormat.IMAGE, "image/jpeg"),
    (b"GIF87a", DocumentFormat.IMAGE, "image/gif"),
    (b"GIF89a", DocumentFormat.IMAGE, "image/gif"),
    (b"II*\x00", DocumentFormat.IMAGE, "image/tiff"),
    (b"MM\x00*", DocumentFormat.IMAGE, "image/tiff"),
    (b"BM", DocumentFormat.IMAGE, "image/bmp"),
]

# File extension mappings
_EXTENSION_MAPPINGS: dict[str, tuple[DocumentFormat, str]] = {
    ".pdf": (DocumentFormat.PDF, "application/pdf"),
    ".txt": (DocumentFormat.TXT, "text/plain"),
    ".text": (DocumentFormat.TXT, "text/plain"),
    ".md": (DocumentFormat.TXT, "text/markdown"),
    ".log": (DocumentFormat.TXT, "text/plain"),
    ".csv": (DocumentFormat.TXT, "text/csv"),
    ".png": (DocumentFormat.IMAGE, "image/png"),
    ".jpg": (DocumentFormat.IMAGE, "image/jpeg"),
    ".jpeg": (DocumentFormat.IMAGE, "image/jpeg"),
    ".tiff": (DocumentFormat.IMAGE, "image/tiff"),
    ".tif": (DocumentFormat.IMAGE, "image/tiff"),
    ".bmp": (DocumentFormat.IMAGE, "image/bmp"),
    ".webp": (DocumentFormat.IMAGE, "image/webp"),
}


class DocumentFormatDetector:
    """Robust multi-layer format detector analyzing magic bytes, file extensions, and text structure."""

    @classmethod
    def detect_format(
        cls,
        source: bytes | str | Path,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> tuple[DocumentFormat, str]:
        """Detect the document format and MIME type.

        Returns:
            Tuple of (DocumentFormat, detected_mime_string).
        """
        # 1. Check if source is a pure string without file path characteristics
        if isinstance(source, str):
            # If string contains newline or doesn't exist on disk as a file, treat as manual text
            p = Path(source)
            if "\n" in source or not p.exists() or len(source) > 260:
                return DocumentFormat.MANUAL_TEXT, "text/plain"
            # It's an existing file path, read bytes
            try:
                data = p.read_bytes()
                fname = filename or p.name
                return cls.detect_from_bytes(data, filename=fname, mime_type=mime_type)
            except Exception:
                return DocumentFormat.MANUAL_TEXT, "text/plain"

        if isinstance(source, Path):
            if source.exists() and source.is_file():
                try:
                    data = source.read_bytes()
                    return cls.detect_from_bytes(data, filename=filename or source.name, mime_type=mime_type)
                except Exception:
                    pass
            fname = filename or source.name
            return cls.detect_from_filename(fname)

        if isinstance(source, bytes):
            return cls.detect_from_bytes(source, filename=filename, mime_type=mime_type)

        return DocumentFormat.UNKNOWN, "application/octet-stream"

    @classmethod
    def detect_from_bytes(
        cls,
        data: bytes,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> tuple[DocumentFormat, str]:
        """Detect document format by inspecting magic byte signatures, then fallback to filename."""
        if not data:
            return DocumentFormat.TXT, "text/plain"

        # Check explicit MIME if provided
        if mime_type:
            mime_lower = mime_type.lower()
            if "pdf" in mime_lower:
                return DocumentFormat.PDF, "application/pdf"
            if any(img in mime_lower for img in ("image/", "png", "jpeg", "jpg", "tiff", "webp", "bmp")):
                return DocumentFormat.IMAGE, mime_lower
            if any(txt in mime_lower for txt in ("text/", "plain", "csv", "markdown")):
                return DocumentFormat.TXT, "text/plain"

        # Magic bytes inspection (header signatures)
        header = data[:32]
        for magic, fmt, detected_mime in _MAGIC_SIGNATURES:
            if header.startswith(magic):
                return fmt, detected_mime

        # Special check for WEBP (RIFF....WEBP)
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return DocumentFormat.IMAGE, "image/webp"

        # Fallback to file extension if filename provided
        if filename:
            fmt, ext_mime = cls.detect_from_filename(filename)
            if fmt != DocumentFormat.UNKNOWN:
                return fmt, ext_mime

        # If no binary magic match, test if valid UTF-8/ASCII text
        try:
            sample = data[:2048]
            sample.decode("utf-8")
            # If decodable without null bytes, it is plain text
            if b"\x00" not in sample:
                return DocumentFormat.TXT, "text/plain"
        except UnicodeDecodeError:
            pass

        return DocumentFormat.UNKNOWN, "application/octet-stream"

    @classmethod
    def detect_from_filename(cls, filename: str) -> tuple[DocumentFormat, str]:
        """Inspect file extension to determine document format."""
        ext = Path(filename).suffix.lower()
        if ext in _EXTENSION_MAPPINGS:
            return _EXTENSION_MAPPINGS[ext]
        return DocumentFormat.UNKNOWN, "application/octet-stream"
