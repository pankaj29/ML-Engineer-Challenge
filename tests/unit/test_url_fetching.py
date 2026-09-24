"""Unit tests for fetching an image by URL.

`image_url` asks our server to make an outbound request on a caller's behalf.
That is Server-Side Request Forgery territory - the classic path from "public
API" to "read the cloud instance's credentials at 169.254.169.254". The audit
found `api/dependencies.py` at 68%, with `fetch_image_url` - the whole 56-line
fetcher - entirely uncovered.

These test the refusals without touching the network (the SSRF guard rejects
before any socket is opened) and stub httpx for the paths that would.
"""

from __future__ import annotations

import httpx
import pytest

from api.dependencies import fetch_image_url
from api.exceptions import ImageTooLargeError, ValidationError


class TestSsrfProtection:
    """Every one of these must be refused before a connection is attempted."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/x.jpg",
            "http://localhost/x.jpg",
            "http://169.254.169.254/latest/meta-data/",  # cloud credentials
            "http://10.0.0.5/x.jpg",
            "http://192.168.1.1/x.jpg",
            "http://172.16.0.1/x.jpg",
            "http://[::1]/x.jpg",
        ],
    )
    async def test_private_and_loopback_addresses_are_refused(self, url: str) -> None:
        with pytest.raises(ValidationError):
            await fetch_image_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/x.jpg",
            "gopher://example.com/x",
            "data:image/png;base64,AAAA",
        ],
    )
    async def test_non_http_schemes_are_refused(self, url: str) -> None:
        with pytest.raises(ValidationError):
            await fetch_image_url(url)

    @pytest.mark.parametrize(
        "url", ["http://example.com:22/x.jpg", "http://example.com:3306/x.jpg"]
    )
    async def test_unexpected_ports_are_refused(self, url: str) -> None:
        """Port 22 and 3306 are not image servers; they are SSH and MySQL."""
        with pytest.raises(ValidationError):
            await fetch_image_url(url)

    async def test_missing_hostname_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            await fetch_image_url("http:///x.jpg")


class _FakeResponse:
    def __init__(self, *, status_code=200, headers=None, chunks=(b"data",)):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks

    async def aiter_bytes(self, _size=None):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    """Stands in for httpx.AsyncClient, recording how it was constructed."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self._response = getattr(type(self), "response", _FakeResponse())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, _method, _url):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture
def stub_httpx(monkeypatch):
    """Replace httpx.AsyncClient with the fake, and hand back the class."""

    def _install(response):
        _FakeClient.response = response
        monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
        return _FakeClient

    return _install


class TestFetching:
    URL = "http://example.com/photo.jpg"

    async def test_returns_the_body(self, stub_httpx) -> None:
        stub_httpx(_FakeResponse(chunks=(b"abc", b"def")))
        assert await fetch_image_url(self.URL) == b"abcdef"

    async def test_redirects_are_disabled(self, stub_httpx) -> None:
        """A public URL that 302s to 169.254.169.254 would defeat the guard."""
        client = stub_httpx(_FakeResponse())
        await fetch_image_url(self.URL)
        assert client.last_kwargs.get("follow_redirects") is False

    async def test_a_timeout_is_configured(self, stub_httpx) -> None:
        client = stub_httpx(_FakeResponse())
        await fetch_image_url(self.URL)
        assert client.last_kwargs.get("timeout")

    async def test_http_error_status_is_reported(self, stub_httpx) -> None:
        stub_httpx(_FakeResponse(status_code=404))
        with pytest.raises(ValidationError, match="404"):
            await fetch_image_url(self.URL)

    async def test_oversized_content_length_is_refused_before_download(self, stub_httpx) -> None:
        stub_httpx(_FakeResponse(headers={"content-length": str(50 * 1024 * 1024)}))
        with pytest.raises(ImageTooLargeError):
            await fetch_image_url(self.URL)

    async def test_a_lying_content_length_is_caught_while_streaming(self, stub_httpx) -> None:
        """The real enforcement: a server can omit or understate the header."""
        big = b"x" * (1024 * 1024)
        stub_httpx(_FakeResponse(headers={}, chunks=tuple(big for _ in range(20))))
        with pytest.raises(ImageTooLargeError):
            await fetch_image_url(self.URL)

    async def test_timeout_becomes_a_clean_validation_error(self, stub_httpx) -> None:
        stub_httpx(httpx.TimeoutException("too slow"))
        with pytest.raises(ValidationError, match=r"[Tt]imed out"):
            await fetch_image_url(self.URL)

    async def test_transport_error_becomes_a_clean_validation_error(self, stub_httpx) -> None:
        """A DNS failure must not surface as a raw httpx traceback."""
        stub_httpx(httpx.ConnectError("no route"))
        with pytest.raises(ValidationError):
            await fetch_image_url(self.URL)
