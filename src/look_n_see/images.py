"""Normalize image inputs (file path, data URI, http(s) URL) to data URIs."""

from __future__ import annotations

import base64
import contextlib
import io
import mimetypes
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

DEFAULT_MEDIA_TYPE = "image/png"
# Hard cap on decoded image bytes; vision providers reject larger payloads anyway.
MAX_IMAGE_BYTES = 25 * 1024 * 1024

_DATA_URI_RE = re.compile(
    r"^data:(?P<media>[^;,]*)(?P<params>[^,]*),(?P<data>.*)$", re.DOTALL | re.IGNORECASE
)
_USERINFO_RE = re.compile(r"(://)[^/@\s]+@")


class ImageInputError(ValueError):
    """Raised when an image input cannot be resolved."""


class _HttpOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that only follows http(s) targets.

    The stdlib allows redirects to ftp://; that scheme escape is outside the
    documented http(s) input contract, so refuse it here.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 - stdlib API
        if urlparse(newurl).scheme.lower() not in ("http", "https"):
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"redirect to disallowed scheme: {urlparse(newurl).scheme}",
                headers,
                io.BytesIO(),  # explicit body: avoids CPython 3.14's fp=None tempfile
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpOnlyRedirectHandler())


def _open_url(url: str, timeout: float):
    """Open one http(s) URL with the scheme-restricting redirect handler."""
    return _OPENER.open(url, timeout=timeout)


def _redact_url(url: str) -> str:
    """Mask embedded credentials (https://user:pass@...) in an echoed URL."""
    return _USERINFO_RE.sub(r"\1***@", url)


def _is_http_url(value: str) -> bool:
    # RFC 3986: URI schemes are case-insensitive; urlparse and the redirect
    # handler already normalize, so "HTTP://..." must count as http(s).
    return value.lower().startswith(("http://", "https://"))


def load_image(image: str) -> tuple[str, bytes]:
    """Resolve `image` to (media_type, bytes).

    Accepts a base64 data URI, an http(s) URL, or a path to a file on disk.
    """
    if image.lower().startswith("data:"):
        return _from_data_uri(image)
    if _is_http_url(image):
        return _from_url(image)
    if os.path.isfile(image):
        return _from_path(image)
    raise ImageInputError(
        f"image input is neither an existing file, a data: URI, nor an http(s) URL: {image!r}"
    )


def to_data_uri(image: str) -> str:
    """Resolve `image` to a base64 data URI for embedding in chat messages."""
    media_type, data = load_image(image)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{b64}"


def _from_data_uri(uri: str) -> tuple[str, bytes]:
    match = _DATA_URI_RE.match(uri)
    if not match:
        raise ImageInputError("malformed data: URI")
    media_type = match.group("media") or DEFAULT_MEDIA_TYPE
    payload = match.group("data")
    # Base64 expands to ~4/3 of the decoded size; reject oversize payloads
    # before allocating.
    if len(payload) > (MAX_IMAGE_BYTES * 4 // 3) + 4:
        raise ImageInputError(f"data: URI payload exceeds {MAX_IMAGE_BYTES} byte limit")
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception:
        try:
            data = base64.urlsafe_b64decode(payload)
        except Exception as exc:
            raise ImageInputError("data: URI payload is not valid base64") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(f"data: URI payload exceeds {MAX_IMAGE_BYTES} byte limit")
    if not data:
        raise ImageInputError("data: URI contains no image bytes")
    return media_type, data


def _from_path(path: str) -> tuple[str, bytes]:
    media_type = mimetypes.guess_type(path)[0] or DEFAULT_MEDIA_TYPE
    try:
        # Read one byte past the cap: a file that grows after a size check is
        # still rejected without allocating it whole.
        with open(path, "rb") as fh:
            data = fh.read(MAX_IMAGE_BYTES + 1)
    except OSError as exc:
        raise ImageInputError(f"could not read image file {path!r}: {exc}") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(f"image file exceeds {MAX_IMAGE_BYTES} byte limit: {path}")
    if not data:
        raise ImageInputError(f"image file is empty: {path}")
    return media_type, data


def _from_url(url: str, timeout: float = 30.0) -> tuple[str, bytes]:
    try:
        with _open_url(url, timeout) as resp:
            data = resp.read(MAX_IMAGE_BYTES + 1)
            media_type = resp.headers.get_content_type()
    except Exception as exc:
        # HTTPError is itself a file-like addinfourl wrapper around the error
        # body (a tempfile when the server sent none); closing the error also
        # releases that body, which a bare fp.close() would leave to the GC.
        if isinstance(exc, urllib.error.HTTPError):
            # A close failure must never mask the descriptive input error.
            with contextlib.suppress(Exception):
                exc.close()
        raise ImageInputError(f"failed to fetch image URL {_redact_url(url)!r}: {exc}") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(f"image URL exceeds {MAX_IMAGE_BYTES} byte limit: {_redact_url(url)!r}")
    if not data:
        raise ImageInputError(f"image URL returned no bytes: {_redact_url(url)!r}")
    if not media_type or media_type == "application/octet-stream":
        media_type = mimetypes.guess_type(urlparse(url).path)[0] or DEFAULT_MEDIA_TYPE
    return media_type, data
