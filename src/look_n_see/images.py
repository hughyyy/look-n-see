"""Normalize image inputs (file path, data URI, http(s) URL) to data URIs."""

from __future__ import annotations

import base64
import contextlib
import io
import ipaddress
import mimetypes
import os
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

DEFAULT_MEDIA_TYPE = "image/png"
# Hard cap on decoded image bytes; vision providers reject larger payloads anyway.
MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_URL_TIMEOUT = 30.0

_DATA_URI_RE = re.compile(
    r"^data:(?P<media>[^;,]*)(?P<params>[^,]*),(?P<data>.*)$", re.DOTALL | re.IGNORECASE
)
_USERINFO_RE = re.compile(r"(://)[^/@\s]+@")
# Presigned URLs carry credentials in the query string; never echo it.
_QUERY_RE = re.compile(r"\?[^#\s]*")


class ImageInputError(ValueError):
    """Raised when an image input cannot be resolved."""


def _host_is_private(url: str) -> bool:
    """True when `url` resolves to a loopback/private/link-local address.

    An unresolvable host is *not* treated as private: the subsequent fetch
    raises the real DNS error and the offline test suite stays hermetic.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:  # pragma: no cover - http(s) URLs always carry a host
        return False
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addresses = {str(ipaddress.ip_address(host))}
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError:
            return False
        addresses = {info[4][0] for info in infos}
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address.split("%")[0])
        except ValueError:  # pragma: no cover - getaddrinfo yields valid IPs
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


class _HttpOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that only follows public http(s) targets.

    The stdlib allows redirects to ftp://; that scheme escape is outside the
    documented http(s) input contract, so refuse it here. Private/loopback
    targets are refused too unless the caller opted in.
    """

    def __init__(self, allow_private: bool = False):
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 - stdlib API
        scheme = urlparse(newurl).scheme.lower()
        if scheme not in ("http", "https"):
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"redirect to disallowed scheme: {scheme}",
                headers,
                io.BytesIO(),  # explicit body: avoids CPython 3.14's fp=None tempfile
            )
        if not self.allow_private and _host_is_private(newurl):
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"redirect to private address blocked: {_redact_url(newurl)!r}",
                headers,
                io.BytesIO(),
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENERS = {
    False: urllib.request.build_opener(_HttpOnlyRedirectHandler(allow_private=False)),
    True: urllib.request.build_opener(_HttpOnlyRedirectHandler(allow_private=True)),
}


def _open_url(url: str, timeout: float, allow_private: bool = False):
    """Open one http(s) URL with the scheme/host-restricting redirect handler."""
    return _OPENERS[allow_private].open(url, timeout=timeout)


def _redact_url(url: str) -> str:
    """Mask embedded credentials and query strings in an echoed URL."""
    masked = _USERINFO_RE.sub(r"\1***@", url)
    return _QUERY_RE.sub("?***", masked)


def _is_http_url(value: str) -> bool:
    # RFC 3986: URI schemes are case-insensitive; urlparse and the redirect
    # handler already normalize, so "HTTP://..." must count as http(s).
    return value.lower().startswith(("http://", "https://"))


def _check_media_type(media_type: str, source: str) -> str:
    if not media_type.lower().startswith("image/"):
        raise ImageInputError(
            f"unsupported media type {media_type!r} from {source}: expected image/*"
        )
    return media_type


def load_image(
    image: str,
    *,
    timeout: float | None = None,
    allow_private_urls: bool = False,
) -> tuple[str, bytes]:
    """Resolve `image` to (media_type, bytes).

    Accepts a base64 data URI, an http(s) URL, or a path to a file on disk.
    Non-image media types are rejected; URL fetches are bounded by `timeout`
    and, by default, restricted to public hosts.
    """
    if image.lower().startswith("data:"):
        media_type, data = _from_data_uri(image)
    elif _is_http_url(image):
        media_type, data = _from_url(
            image,
            timeout=timeout if timeout is not None else DEFAULT_URL_TIMEOUT,
            allow_private_urls=allow_private_urls,
        )
    elif os.path.isfile(image):
        media_type, data = _from_path(image)
    else:
        raise ImageInputError(
            f"image input is neither an existing file, a data: URI, nor an http(s) URL: {image!r}"
        )
    return _check_media_type(media_type, "image input"), data


def to_data_uri(
    image: str,
    *,
    timeout: float | None = None,
    allow_private_urls: bool = False,
) -> str:
    """Resolve `image` to a base64 data URI for embedding in chat messages."""
    media_type, data = load_image(
        image, timeout=timeout, allow_private_urls=allow_private_urls
    )
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
        # Lenient fallback: accepts urlsafe alphabet and line-wrapped payloads.
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


def _from_url(
    url: str,
    timeout: float = DEFAULT_URL_TIMEOUT,
    allow_private_urls: bool = False,
) -> tuple[str, bytes]:
    if not allow_private_urls and _host_is_private(url):
        raise ImageInputError(
            f"refusing to fetch image URL on a private address: {_redact_url(url)!r} "
            "(set VISION_ALLOW_PRIVATE_URLS=1 to allow)"
        )
    try:
        with _open_url(url, timeout, allow_private_urls) as resp:
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
