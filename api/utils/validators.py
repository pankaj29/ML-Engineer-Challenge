"""Input validation — the security boundary of the API.

Plain English:
    Everything a user uploads is treated as hostile until proven otherwise.
    This module answers one question: "is this actually a safe, usable image?"
    It runs *before* any pixel reaches a model.

Four classes of attack are handled explicitly:

1. **Oversized uploads** — a 2 GB file would exhaust memory. Rejected on byte
   count before we even try to decode.
2. **Decompression bombs** — a 4 KB PNG can legally expand to 100,000 x
   100,000 pixels (~40 GB of RAM). We read the header first and check the
   declared pixel count *before* decoding the pixels.
3. **Content spoofing** — a file named ``cat.jpg`` may actually contain a ZIP
   or an SVG with embedded JavaScript. We identify format from the magic bytes
   Pillow reads, never from the filename or the client's Content-Type.
4. **SSRF via image_url** — a URL like ``http://169.254.169.254/`` would make
   our server fetch cloud credentials on the attacker's behalf. We resolve the
   hostname and refuse private, loopback and link-local addresses.
"""

from __future__ import annotations

import io
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from api.config import Settings, settings
from api.exceptions import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedFormatError,
    ValidationError,
)
from api.logging_config import get_logger

logger = get_logger(__name__)

# Pillow's own bomb guard. We set it from config and additionally do our own
# explicit check so that we can return a helpful error instead of a warning.
Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

# Magic byte prefixes for the formats we accept. Used as a cheap first-pass
# filter before handing bytes to Pillow.
_MAGIC_PREFIXES: dict[str, tuple[bytes, ...]] = {
    "JPEG": (b"\xff\xd8\xff",),
    "PNG": (b"\x89PNG\r\n\x1a\n",),
    "WEBP": (b"RIFF",),  # bytes 8-12 must also be "WEBP"; checked below
    "BMP": (b"BM",),
    "GIF": (b"GIF87a", b"GIF89a"),
    "TIFF": (b"II*\x00", b"MM\x00*"),
}


@dataclass(frozen=True)
class ImageMetadata:
    """Facts about a validated image, returned to callers for logging."""

    format: str
    width: int
    height: int
    mode: str
    size_bytes: int
    has_alpha: bool

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


def sniff_format(data: bytes) -> str | None:
    """Identify an image format from its magic bytes.

    Returns the format name, or ``None`` if the bytes match nothing we know.
    This never trusts a filename or a client-supplied MIME type.
    """
    if len(data) < 12:
        return None
    for fmt, prefixes in _MAGIC_PREFIXES.items():
        for prefix in prefixes:
            if data.startswith(prefix):
                # WEBP shares the generic RIFF container with other formats,
                # so confirm the sub-type at offset 8.
                if fmt == "WEBP" and data[8:12] != b"WEBP":
                    continue
                return fmt
    return None


def validate_image_bytes(
    data: bytes,
    *,
    config: Settings | None = None,
    field_name: str = "image",
) -> ImageMetadata:
    """Validate raw image bytes and return their metadata.

    The checks run cheapest-first, so a hostile upload is rejected as early
    and as inexpensively as possible.

    Args:
        data: Raw bytes as uploaded.
        config: Settings to validate against. Defaults to the global settings;
            injectable so tests can tighten the limits without patching globals.
        field_name: Name used in error messages, e.g. ``"items[3].image"``.

    Returns:
        :class:`ImageMetadata` describing the validated image.

    Raises:
        InvalidImageError: Bytes are empty, truncated or not decodable.
        ImageTooLargeError: Exceeds the byte, pixel or dimension budget.
        UnsupportedFormatError: Decodes fine but the format is not allowed.
    """
    cfg = config or settings

    # --- 1. Non-empty --------------------------------------------------
    if not data:
        raise InvalidImageError(
            f"The {field_name} field contained no data.",
            details={"field": field_name},
        )

    # --- 2. Byte budget (cheapest possible check) ----------------------
    size = len(data)
    if size > cfg.max_image_bytes:
        raise ImageTooLargeError(
            (
                f"The {field_name} is {size / 1_048_576:.1f} MB, which exceeds "
                f"the {cfg.max_image_bytes / 1_048_576:.0f} MB limit."
            ),
            details={
                "field": field_name,
                "size_bytes": size,
                "limit_bytes": cfg.max_image_bytes,
            },
        )

    # --- 3. Magic bytes -------------------------------------------------
    sniffed = sniff_format(data)
    if sniffed is None:
        raise InvalidImageError(
            f"The {field_name} does not look like an image file.",
            details={"field": field_name, "size_bytes": size},
            internal_message=f"unrecognised magic bytes: {data[:12]!r}",
        )

    # --- 4. Header parse, BEFORE decoding pixels ------------------------
    # Pillow reads only the header here, so a decompression bomb is caught
    # without ever allocating its full pixel buffer.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or sniffed).upper()
            width, height = probe.size
            mode = probe.mode
            # Captured inside the context manager: `probe` is closed on exit.
            has_transparency = "transparency" in probe.info
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            f"The {field_name} could not be decoded as an image.",
            details={"field": field_name, "sniffed_format": sniffed},
            internal_message=str(exc),
        ) from exc
    except Exception as exc:  # Pillow raises a wide variety of parse errors
        raise InvalidImageError(
            f"The {field_name} appears to be corrupt or truncated.",
            details={"field": field_name},
            internal_message=f"{type(exc).__name__}: {exc}",
        ) from exc

    # --- 5. Format allow-list ------------------------------------------
    allowed = cfg.allowed_formats_set
    if fmt not in allowed:
        raise UnsupportedFormatError(
            f"{fmt} images are not supported. Allowed formats: {', '.join(sorted(allowed))}.",
            details={"field": field_name, "format": fmt, "allowed": sorted(allowed)},
        )

    # --- 6. Dimension sanity -------------------------------------------
    if width < cfg.min_image_dimension or height < cfg.min_image_dimension:
        raise InvalidImageError(
            (
                f"The {field_name} is {width}x{height} pixels, which is smaller "
                f"than the {cfg.min_image_dimension}x{cfg.min_image_dimension} minimum."
            ),
            details={"field": field_name, "width": width, "height": height},
        )
    if width > cfg.max_image_dimension or height > cfg.max_image_dimension:
        raise ImageTooLargeError(
            (
                f"The {field_name} is {width}x{height} pixels, which exceeds the "
                f"{cfg.max_image_dimension} pixel limit on either side."
            ),
            details={"field": field_name, "width": width, "height": height},
        )

    # --- 7. Decompression-bomb guard -----------------------------------
    pixels = width * height
    if pixels > cfg.max_image_pixels:
        raise ImageTooLargeError(
            (
                f"The {field_name} declares {pixels:,} pixels, above the "
                f"{cfg.max_image_pixels:,} limit. This is refused as a possible "
                "decompression bomb."
            ),
            details={
                "field": field_name,
                "pixels": pixels,
                "limit_pixels": cfg.max_image_pixels,
                "compression_ratio": round(pixels / max(size, 1), 1),
            },
        )

    # --- 8. Integrity -----------------------------------------------------
    # The header parse above deliberately does NOT decode pixels, so that a
    # decompression bomb is caught before it can allocate memory. The cost is
    # that a truncated file passes the header check. Pillow's verify() walks
    # the chunk structure and checks CRCs without building the pixel buffer,
    # so it catches truncation and corruption cheaply. It must run on a fresh
    # Image object, and it invalidates that object, hence the second open().
    try:
        with Image.open(io.BytesIO(data)) as check:
            check.verify()
    except Exception as exc:
        raise InvalidImageError(
            f"The {field_name} is corrupt or truncated.",
            details={"field": field_name, "format": fmt},
            internal_message=f"verify() failed: {type(exc).__name__}: {exc}",
        ) from exc

    return ImageMetadata(
        format=fmt,
        width=width,
        height=height,
        mode=mode,
        size_bytes=size,
        has_alpha=mode in ("RGBA", "LA", "PA") or has_transparency,
    )


def validate_batch_size(count: int, tier: str, *, config: Settings | None = None) -> None:
    """Check a batch against the caller's tier allowance.

    Free-tier users get a small cap so one caller cannot monopolise the worker
    pool; paid tiers get progressively more.
    """
    cfg = config or settings
    tier_caps = {
        "free": min(5, cfg.max_batch_size),
        "basic": min(16, cfg.max_batch_size),
        "pro": min(32, cfg.max_batch_size),
        "enterprise": cfg.max_batch_size,
    }
    cap = tier_caps.get(tier.lower(), tier_caps["free"])
    if count > cap:
        from api.exceptions import BatchTooLargeError

        raise BatchTooLargeError(
            f"Your '{tier}' tier allows up to {cap} images per batch; you sent {count}.",
            details={"submitted": count, "limit": cap, "tier": tier},
        )


# ---------------------------------------------------------------------------
# URL validation (SSRF defence)
# ---------------------------------------------------------------------------
def _is_blocked_ip(ip_str: str) -> bool:
    """True if an address is one we must never let the server fetch."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable: refuse
    return bool(
        ip.is_private  # 10.x, 192.168.x, 172.16-31.x
        or ip.is_loopback  # 127.x — our own services
        or ip.is_link_local  # 169.254.x — cloud metadata endpoints
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_image_url(url: str, *, allow_private: bool = False) -> str:
    """Validate an ``image_url`` before the server fetches it.

    SECURITY: without this check, ``image_url`` would let any caller make our
    server issue arbitrary requests from inside the production network — the
    classic SSRF escalation path to cloud instance credentials at
    ``169.254.169.254``.

    Args:
        url: The URL to validate.
        allow_private: Set True only in tests, where the fixture server runs
            on localhost.

    Returns:
        The URL, unchanged, if it is safe to fetch.

    Raises:
        ValidationError: Scheme, host or resolved address is not permitted.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValidationError(
            "image_url must use http:// or https://.",
            details={"scheme": parsed.scheme},
        )
    if not parsed.hostname:
        raise ValidationError("image_url has no hostname.", details={"url": url[:100]})

    # The test-only escape hatch short-circuits every remaining check, so a
    # fixture server on an arbitrary localhost port is reachable.
    if allow_private:
        return url

    if parsed.port is not None and parsed.port not in (80, 443, 8080, 8443):
        raise ValidationError(
            "image_url may only target standard HTTP(S) ports.",
            details={"port": parsed.port},
        )

    # Resolve every address the hostname maps to and reject if ANY is private.
    # Checking all of them defends against DNS round-robin tricks where one
    # answer is public and the next is 127.0.0.1.
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValidationError(
            "The hostname in image_url could not be resolved.",
            details={"hostname": parsed.hostname},
            internal_message=str(exc),
        ) from exc

    for info in infos:
        ip_str = str(info[4][0])
        if _is_blocked_ip(ip_str):
            logger.warning(
                "ssrf_attempt_blocked",
                extra={"hostname": parsed.hostname, "resolved_ip": ip_str},
            )
            raise ValidationError(
                "image_url resolves to a private or reserved address, which is not allowed.",
                details={"hostname": parsed.hostname},
            )

    return url


def validate_model_selection(name: str | None, version: str | None) -> tuple[str | None, str]:
    """Normalise a model name/version pair from a request.

    Returns ``(name, version)`` with version defaulting to ``"latest"``.
    """
    if name is not None and not name.replace("-", "").replace("_", "").isalnum():
        raise ValidationError(
            "model_name may only contain letters, digits, hyphens and underscores.",
            details={"model_name": name},
        )
    return name, version or "latest"


__all__ = [
    "ImageMetadata",
    "sniff_format",
    "validate_batch_size",
    "validate_image_bytes",
    "validate_image_url",
    "validate_model_selection",
]
