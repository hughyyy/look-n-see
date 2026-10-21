"""Normalize image inputs (file path, data URI, http(s) URL) to data URIs."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import urllib.request
from urllib.parse import urlparse

DEFAULT_MEDIA_TYPE = "image/png"

_DATA_URI_RE = re.compile(r"^data:(?P<media>[^;,]*)(?P<params>[^,]*),(?P<data>.*)$", re.DOTALL)


class ImageInputError(ValueError):
    """Raised when an image input cannot be resolved."""


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def load_image(image: str) -> tuple[str, bytes]:
    """Resolve `image` to (media_type, bytes).

    Accepts a base64 data URI, an http(s) URL, or a path to a file on disk.
    """
    if image.startswith("data:"):
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
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception:
        try:
            data = base64.urlsafe_b64decode(payload)
        except Exception as exc:
            raise ImageInputError("data: URI payload is not valid base64") from exc
    return media_type, data


def _from_path(path: str) -> tuple[str, bytes]:
    media_type = mimetypes.guess_type(path)[0] or DEFAULT_MEDIA_TYPE
    with open(path, "rb") as fh:
        return media_type, fh.read()


def _from_url(url: str, timeout: float = 30.0) -> tuple[str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
            media_type = resp.headers.get_content_type()
    except Exception as exc:
        raise ImageInputError(f"failed to fetch image URL {url!r}: {exc}") from exc
    if not media_type or media_type == "application/octet-stream":
        media_type = mimetypes.guess_type(urlparse(url).path)[0] or DEFAULT_MEDIA_TYPE
    return media_type, data
