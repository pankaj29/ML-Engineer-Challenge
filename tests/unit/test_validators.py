"""Unit tests for input validation.

These are the security boundary tests. Each one encodes an attack or a
malformed input that must be rejected, so a future refactor cannot quietly
reopen the hole.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from api.config import Settings
from api.exceptions import (
    BatchTooLargeError,
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedFormatError,
    ValidationError,
)
from api.utils.validators import (
    sniff_format,
    validate_batch_size,
    validate_image_bytes,
    validate_image_url,
    validate_model_selection,
)
from tests.conftest import make_image, make_noise_image


class TestSniffFormat:
    """Format detection must come from the bytes, never a filename."""

    def test_detects_png(self) -> None:
        assert sniff_format(make_image(fmt="PNG")) == "PNG"

    def test_detects_jpeg(self) -> None:
        assert sniff_format(make_image(fmt="JPEG")) == "JPEG"

    def test_detects_webp(self) -> None:
        assert sniff_format(make_image(fmt="WEBP")) == "WEBP"

    def test_detects_bmp(self) -> None:
        assert sniff_format(make_image(fmt="BMP")) == "BMP"

    def test_detects_gif(self) -> None:
        assert sniff_format(make_image(fmt="GIF")) == "GIF"

    def test_rejects_text(self) -> None:
        assert sniff_format(b"just some plain text here") is None

    def test_rejects_too_short(self) -> None:
        assert sniff_format(b"\x89PNG") is None

    def test_riff_that_is_not_webp(self) -> None:
        """A RIFF container that is not WEBP (e.g. a WAV) must not pass as WEBP."""
        wav = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 20
        assert sniff_format(wav) is None


class TestValidateImageBytes:
    """The main validation entry point."""

    def test_accepts_valid_png(self, sample_image: bytes) -> None:
        meta = validate_image_bytes(sample_image)
        assert meta.format == "PNG"
        assert meta.width == 224
        assert meta.height == 224
        assert meta.pixels == 224 * 224
        assert meta.size_bytes == len(sample_image)

    def test_accepts_valid_jpeg(self, sample_jpeg: bytes) -> None:
        assert validate_image_bytes(sample_jpeg).format == "JPEG"

    def test_rejects_empty(self) -> None:
        with pytest.raises(InvalidImageError) as exc:
            validate_image_bytes(b"")
        assert exc.value.code == "INVALID_IMAGE"

    def test_rejects_non_image(self, not_an_image: bytes) -> None:
        with pytest.raises(InvalidImageError):
            validate_image_bytes(not_an_image)

    def test_rejects_truncated(self, corrupt_image: bytes) -> None:
        with pytest.raises(InvalidImageError):
            validate_image_bytes(corrupt_image)

    def test_rejects_oversized_bytes(self, strict_settings: Settings) -> None:
        big = make_noise_image(600, 600)
        assert len(big) > strict_settings.max_image_bytes
        with pytest.raises(ImageTooLargeError) as exc:
            validate_image_bytes(big, config=strict_settings)
        assert exc.value.details["limit_bytes"] == strict_settings.max_image_bytes

    def test_rejects_disallowed_format(self, strict_settings: Settings) -> None:
        webp = make_image(fmt="WEBP")
        with pytest.raises(UnsupportedFormatError) as exc:
            validate_image_bytes(webp, config=strict_settings)
        assert "WEBP" in exc.value.details["format"]

    def test_rejects_too_small(self, tiny_image: bytes) -> None:
        with pytest.raises(InvalidImageError):
            validate_image_bytes(tiny_image)

    def test_rejects_too_large_dimension(self, strict_settings: Settings) -> None:
        wide = make_image(width=2000, height=64)
        with pytest.raises(ImageTooLargeError):
            validate_image_bytes(wide, config=strict_settings)

    def test_rejects_decompression_bomb(self) -> None:
        """A small file declaring an enormous pixel count must be refused.

        A flat-colour PNG compresses to a few kilobytes no matter how large it
        claims to be. Decoding one would allocate gigabytes.
        """
        config = Settings(environment="test", max_image_pixels=10_000, max_image_dimension=100_000)
        buffer = io.BytesIO()
        Image.new("RGB", (4000, 4000), (255, 255, 255)).save(buffer, "PNG")
        bomb = buffer.getvalue()

        # The premise of the test: the file really is small.
        assert len(bomb) < 200_000

        with pytest.raises(ImageTooLargeError) as exc:
            validate_image_bytes(bomb, config=config)
        assert exc.value.details["pixels"] == 16_000_000

    def test_grayscale_is_accepted(self, grayscale_image: bytes) -> None:
        """Greyscale is a legitimate image; conversion happens in preprocessing."""
        meta = validate_image_bytes(grayscale_image)
        assert meta.mode == "L"

    def test_rgba_reports_alpha(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(buffer, "PNG")
        assert validate_image_bytes(buffer.getvalue()).has_alpha is True

    def test_error_message_names_the_field(self) -> None:
        with pytest.raises(InvalidImageError) as exc:
            validate_image_bytes(b"", field_name="items[3].image")
        assert "items[3].image" in exc.value.message


class TestValidateImageUrl:
    """SSRF defence. Each case is a real escalation path."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",  # AWS instance metadata
            "http://metadata.google.internal/",  # GCP metadata
            "http://127.0.0.1:6379/",  # our own Redis
            "http://localhost:5432/",  # our own Postgres
            "http://10.0.0.1/admin",  # private range
            "http://192.168.1.1/",  # private range
            "http://172.16.0.1/",  # private range
            "http://[::1]/",  # IPv6 loopback
            "http://0.0.0.0/",  # unspecified
        ],
    )
    def test_blocks_internal_addresses(self, url: str) -> None:
        with pytest.raises(ValidationError):
            validate_image_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "gopher://example.com/",
            "ftp://example.com/x.png",
            "data:image/png;base64,AAAA",
        ],
    )
    def test_blocks_non_http_schemes(self, url: str) -> None:
        with pytest.raises(ValidationError):
            validate_image_url(url)

    def test_blocks_unusual_ports(self) -> None:
        with pytest.raises(ValidationError) as exc:
            validate_image_url("http://example.com:22/x.png")
        assert exc.value.details["port"] == 22

    def test_allows_private_when_explicitly_enabled(self) -> None:
        """The test-only escape hatch, used by the local fixture server."""
        assert validate_image_url("http://127.0.0.1:8000/x.png", allow_private=True)


class TestValidateBatchSize:
    """Per-tier batch caps."""

    @pytest.mark.parametrize(
        ("tier", "allowed", "rejected"),
        [("free", 5, 6), ("basic", 16, 17), ("pro", 32, 33), ("enterprise", 64, 65)],
    )
    def test_tier_limits(self, tier: str, allowed: int, rejected: int) -> None:
        validate_batch_size(allowed, tier)  # must not raise
        with pytest.raises(BatchTooLargeError):
            validate_batch_size(rejected, tier)

    def test_unknown_tier_gets_free_limit(self) -> None:
        """An unrecognised tier must fail closed, to the smallest allowance."""
        with pytest.raises(BatchTooLargeError) as exc:
            validate_batch_size(6, "platinum-deluxe")
        assert exc.value.details["limit"] == 5


class TestValidateModelSelection:
    def test_defaults_version_to_latest(self) -> None:
        assert validate_model_selection("resnet50", None) == ("resnet50", "latest")

    def test_passes_through_explicit_version(self) -> None:
        assert validate_model_selection("resnet50", "2.1.0") == ("resnet50", "2.1.0")

    def test_allows_hyphens_and_underscores(self) -> None:
        assert validate_model_selection("resnet50-embed_v2", None)[0] == "resnet50-embed_v2"

    @pytest.mark.parametrize("name", ["../../etc/passwd", "model;drop table", "a b"])
    def test_rejects_path_and_injection_characters(self, name: str) -> None:
        with pytest.raises(ValidationError):
            validate_model_selection(name, None)
