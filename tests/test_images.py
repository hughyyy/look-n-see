"""Unit tests for image input parsing."""

import base64
import gc
import http.server
import io
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from unittest import mock

from look_n_see import images
from look_n_see.images import ImageInputError, _HttpOnlyRedirectHandler, load_image, to_data_uri

PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
PNG_DATA_URI = f"data:image/png;base64,{PNG_1PX}"


class DataUriTests(unittest.TestCase):
    def test_roundtrip(self):
        self.assertEqual(to_data_uri(PNG_DATA_URI), PNG_DATA_URI)

    def test_defaults_media_type_when_missing(self):
        media, data = load_image(f"data:;base64,{PNG_1PX}")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, base64.b64decode(PNG_1PX))

    def test_media_type_with_parameters(self):
        # data:image/png;charset=utf-8;base64,... keeps the media type and
        # ignores the parameters when building the final URI.
        uri = f"data:image/png;charset=utf-8;base64,{PNG_1PX}"
        media, data = load_image(uri)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, base64.b64decode(PNG_1PX))
        self.assertEqual(to_data_uri(uri), PNG_DATA_URI)

    def test_rejects_garbage_payload(self):
        with self.assertRaises(ImageInputError):
            load_image("data:image/png;base64,@@@not-base64@@@")

    def test_rejects_oversize_payload(self):
        with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
            big = base64.b64encode(b"x" * 2048).decode("ascii")
            with self.assertRaisesRegex(ImageInputError, "exceeds 1024 byte limit"):
                load_image(f"data:image/png;base64,{big}")

    def test_rejects_malformed_uri(self):
        # "data:" without the comma separator fails the URI grammar before
        # any base64 decoding.
        with self.assertRaisesRegex(ImageInputError, "malformed data: URI"):
            load_image("data:image/png;base64")

    def test_accepts_urlsafe_base64(self):
        # Standard b64decode(validate=True) rejects '-'/'_'; the fallback
        # decoder must accept them.
        payload = base64.urlsafe_b64encode(b"\xfb\xef\x01").decode("ascii")
        media, data = load_image(f"data:image/png;base64,{payload}")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, b"\xfb\xef\x01")

    def test_rejects_oversize_decoded_payload(self):
        # Payload length sits under the pre-decode cap, but decoding pushes
        # the bytes past it — the post-decode check must still fire.
        with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
            payload = base64.b64encode(b"x" * 1026).decode("ascii")
            self.assertLessEqual(len(payload), (1024 * 4 // 3) + 4)
            with self.assertRaisesRegex(ImageInputError, "exceeds 1024 byte limit"):
                load_image(f"data:image/png;base64,{payload}")

    def test_accepts_uppercase_scheme(self):
        # RFC 3986: schemes are case-insensitive, so DATA: is the same scheme.
        uri = "DATA:image/png;base64," + PNG_1PX
        media, data = load_image(uri)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, base64.b64decode(PNG_1PX))

    def test_accepts_line_wrapped_payload(self):
        # Wrapped base64 (newlines) fails strict decoding; the lenient
        # fallback decoder must still recover the bytes.
        wrapped = "\n".join([PNG_1PX[:20], PNG_1PX[20:]])
        _, data = load_image(f"data:image/png;base64,{wrapped}")
        self.assertEqual(data, base64.b64decode(PNG_1PX))

    def test_accepts_exact_byte_limit(self):
        payload = base64.b64encode(b"x" * 1024).decode("ascii")
        with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
            _, data = load_image(f"data:image/png;base64,{payload}")
        self.assertEqual(len(data), 1024)

    def test_rejects_empty_payload(self):
        with self.assertRaisesRegex(ImageInputError, "contains no image bytes"):
            load_image("data:image/png;base64,")

    def test_params_without_media_type(self):
        media, data = load_image(f"data:;charset=utf-8;base64,{PNG_1PX}")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, base64.b64decode(PNG_1PX))

    def test_uppercase_scheme_and_base64_marker(self):
        media, data = load_image("DATA:IMAGE/PNG;BASE64," + PNG_1PX)
        self.assertEqual(media.lower(), "image/png")
        self.assertEqual(data, base64.b64decode(PNG_1PX))


class FilePathTests(unittest.TestCase):
    def test_loads_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.png"
            path.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
            media, data = load_image(str(path))
            self.assertEqual(media, "image/png")
            self.assertEqual(data, bytes.fromhex("89504e470d0a1a0a"))

    def test_rejects_oversize_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.png"
            path.write_bytes(b"x" * 2048)
            with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
                with self.assertRaisesRegex(ImageInputError, "exceeds 1024 byte limit"):
                    load_image(str(path))

    def test_unknown_extension_defaults_media_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot"  # no extension -> mimetypes returns None
            path.write_bytes(b"raw")
            media, data = load_image(str(path))
            self.assertEqual(media, "image/png")  # DEFAULT_MEDIA_TYPE fallback
            self.assertEqual(data, b"raw")

    def test_accepts_exact_byte_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exact.png"
            path.write_bytes(b"x" * 1024)
            with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
                _, data = load_image(str(path))
        self.assertEqual(len(data), 1024)

    def test_to_data_uri_from_path_uses_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.jpg"
            path.write_bytes(b"\xff\xd8\xff\xe0")
            self.assertEqual(
                to_data_uri(str(path)), "data:image/jpeg;base64,/9j/4A=="
            )

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.png"
            path.write_bytes(b"")
            with self.assertRaisesRegex(ImageInputError, "is empty"):
                load_image(str(path))

    def test_unreadable_file_raises_image_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secret.png"
            path.write_bytes(b"x")
            with mock.patch("builtins.open", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(ImageInputError, "could not read image file"):
                    load_image(str(path))

    def test_file_growing_past_cap_is_rejected(self):
        # The read is bounded, so a file that grew after being listed cannot
        # slip past the size cap or allocate the whole payload.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grow.png"
            path.write_bytes(b"x")
            with mock.patch("builtins.open", return_value=io.BytesIO(b"x" * 2048)):
                with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
                    with self.assertRaisesRegex(ImageInputError, "exceeds 1024 byte limit"):
                        load_image(str(path))


class UrlTests(unittest.TestCase):
    def test_fetches_and_uses_content_type(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "image/jpeg"
        fake.read.return_value = b"\xff\xd8\xff\xe0"
        with mock.patch("look_n_see.images._open_url", return_value=fake) as opener:
            media, data = load_image("https://example.com/shot.jpg")
        opener.assert_called_once_with("https://example.com/shot.jpg", 30.0)
        self.assertEqual(media, "image/jpeg")
        self.assertEqual(data, b"\xff\xd8\xff\xe0")

    def test_fetch_failure_raises(self):
        with mock.patch("look_n_see.images._open_url", side_effect=OSError("boom")):
            with self.assertRaises(ImageInputError):
                load_image("https://example.com/missing.png")

    def test_read_failure_mid_stream_raises(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "image/png"
        fake.read.side_effect = OSError("connection reset")
        with mock.patch("look_n_see.images._open_url", return_value=fake):
            with self.assertRaisesRegex(ImageInputError, "failed to fetch"):
                load_image("https://example.com/shot.png")

    def test_fetch_failure_redacts_url_credentials(self):
        with mock.patch("look_n_see.images._open_url", side_effect=OSError("boom")):
            with self.assertRaisesRegex(ImageInputError, r"https://\*\*\*@example.com"):
                load_image("https://user:s3cret@example.com/shots/missing.png")
            with self.assertRaises(ImageInputError) as ctx:
                load_image("https://user:s3cret@example.com/shots/missing.png")
        self.assertNotIn("s3cret", str(ctx.exception))

    def test_rejects_oversize_response(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "image/png"
        fake.read.return_value = b"x" * 2048
        with mock.patch("look_n_see.images.MAX_IMAGE_BYTES", 1024):
            with mock.patch("look_n_see.images._open_url", return_value=fake):
                with self.assertRaisesRegex(ImageInputError, "exceeds 1024 byte limit"):
                    load_image("https://example.com/big.png")

    def test_query_string_extension_still_guessed(self):
        # Content-Type is generic; the guessed type must come from the URL path,
        # which ignores the query string.
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "application/octet-stream"
        fake.read.return_value = b"\x89PNG"
        with mock.patch("look_n_see.images._open_url", return_value=fake):
            media, _ = load_image("https://example.com/shot.png?token=abc")
        self.assertEqual(media, "image/png")

    def test_oversize_error_redacts_credentials(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "image/png"
        fake.read.return_value = b"x" * 2048
        with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
            with mock.patch("look_n_see.images._open_url", return_value=fake):
                with self.assertRaises(ImageInputError) as ctx:
                    load_image("https://user:s3cret@example.com/big.png")
        self.assertNotIn("s3cret", str(ctx.exception))

    def test_redirect_to_non_http_schemes_is_blocked(self):
        handler = _HttpOnlyRedirectHandler()
        req = urllib.request.Request("https://example.com/a.png")
        for target in (
            "ftp://internal/steal.png",
            "file:///etc/passwd",
            "data:text/plain,x",
        ):
            with self.subTest(target=target):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    handler.redirect_request(req, None, 302, "Found", {}, target)
                ctx.exception.close()

    def test_redirect_to_http_is_followed(self):
        handler = _HttpOnlyRedirectHandler()
        req = urllib.request.Request("https://example.com/a.png")
        # The handler delegates to the stdlib, which raises for malformed
        # targets with no scheme; a plain http target must not raise here.
        try:
            handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.com/b.png")
        except urllib.error.HTTPError:
            self.fail("https redirect should be delegated, not blocked")

    def test_octet_stream_falls_back_to_path_guess(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "application/octet-stream"
        fake.read.return_value = b"\x89PNG"
        with mock.patch("look_n_see.images._open_url", return_value=fake):
            media, data = load_image("https://example.com/shot.png")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, b"\x89PNG")

    def test_missing_content_type_falls_back_to_path_guess(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = ""
        fake.read.return_value = b"data"
        with mock.patch("look_n_see.images._open_url", return_value=fake):
            media, _ = load_image("https://example.com/shot.jpeg")
        self.assertEqual(media, "image/jpeg")

    def test_http_error_fp_is_closed(self):
        fp = mock.MagicMock()
        err = urllib.error.HTTPError(
            "https://example.com/gone.png", 404, "Not Found", {}, fp
        )
        with mock.patch("look_n_see.images._open_url", side_effect=err):
            with self.assertRaises(ImageInputError):
                load_image("https://example.com/gone.png")
        fp.close.assert_called_once()

    def test_accepts_exact_byte_limit(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "image/png"
        fake.read.return_value = b"x" * 1024
        with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
            with mock.patch("look_n_see.images._open_url", return_value=fake):
                _, data = load_image("https://example.com/exact.png")
        self.assertEqual(len(data), 1024)

    def test_rejects_empty_body(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "image/png"
        fake.read.return_value = b""
        with mock.patch("look_n_see.images._open_url", return_value=fake):
            with self.assertRaisesRegex(ImageInputError, "returned no bytes"):
                load_image("https://example.com/empty.png")

    def test_http_error_wrapper_is_closed(self):
        # HTTPError is itself a file-like addinfourl wrapper; closing the error
        # releases the body tempfile instead of leaving it to GC.
        err = urllib.error.HTTPError(
            "https://example.com/gone.png", 404, "Not Found", {}, io.BytesIO(b"")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with mock.patch("look_n_see.images._open_url", side_effect=err):
                with self.assertRaises(ImageInputError):
                    load_image("https://example.com/gone.png")
            del err
            gc.collect()
        leaked = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(leaked, [])


class _ImageHandler(http.server.BaseHTTPRequestHandler):
    """Serves PNG bytes, a 302 redirect, and 404s on loopback."""

    PNG_BYTES = bytes.fromhex("89504e470d0a1a0a")

    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path == "/shot.png":
            body = self.PNG_BYTES
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/empty.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/move.png":
            self.send_response(302)
            self.send_header("Location", "/shot.png")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/ftpmove.png":
            self.send_response(302)
            self.send_header("Location", "ftp://internal/steal.png")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/loop.png":
            self.send_response(302)
            self.send_header("Location", "/loop.png")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/filemove.png":
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args):  # keep test output clean
        pass


class LocalServerTests(unittest.TestCase):
    """Exercise the real urllib opener + redirect handler over loopback (offline)."""

    @classmethod
    def setUpClass(cls):
        cls._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()
        cls.port = cls._server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls._server.shutdown()
        cls._server.server_close()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_real_fetch(self):
        media, data = load_image(self._url("/shot.png"))
        self.assertEqual(media, "image/png")
        self.assertEqual(data, _ImageHandler.PNG_BYTES)

    def test_uppercase_scheme_accepted(self):
        # RFC 3986: schemes are case-insensitive; the fetcher must not reject
        # "HTTP://..." inputs.
        media, data = load_image(self._url("/shot.png").replace("http://", "HTTP://"))
        self.assertEqual(media, "image/png")
        self.assertEqual(data, _ImageHandler.PNG_BYTES)

    def test_real_redirect_followed(self):
        media, data = load_image(self._url("/move.png"))
        self.assertEqual(media, "image/png")
        self.assertEqual(data, _ImageHandler.PNG_BYTES)

    def test_real_http_error_raises(self):
        with self.assertRaises(ImageInputError):
            load_image(self._url("/missing.png"))

    def test_ftp_redirect_blocked_over_opener(self):
        # The server answers with an ftp:// Location; the real opener's
        # scheme-restricting redirect handler must refuse it end-to-end.
        with self.assertRaises(ImageInputError):
            load_image(self._url("/ftpmove.png"))

    def test_redirect_loop_raises_image_input_error(self):
        # The stdlib caps redirects; the loop must surface as a tool error, not
        # hang or leak the redirect count.
        with self.assertRaises(ImageInputError):
            load_image(self._url("/loop.png"))

    def test_file_redirect_blocked_over_opener(self):
        with self.assertRaises(ImageInputError):
            load_image(self._url("/filemove.png"))

    def test_empty_body_rejected(self):
        with self.assertRaisesRegex(ImageInputError, "returned no bytes"):
            load_image(self._url("/empty.png"))

    def test_http_error_leaves_no_resource_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with self.assertRaises(ImageInputError):
                load_image(self._url("/missing.png"))
            gc.collect()
        leaked = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(leaked, [])

    def test_blocked_redirect_leaves_no_resource_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with self.assertRaises(ImageInputError):
                load_image(self._url("/ftpmove.png"))
            gc.collect()
        leaked = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(leaked, [])


class InvalidInputTests(unittest.TestCase):
    def test_nonexistent_path_raises(self):
        with self.assertRaises(ImageInputError):
            load_image("/definitely/not/a/real/file.png")

    def test_empty_string_raises(self):
        with self.assertRaisesRegex(ImageInputError, "neither an existing file"):
            load_image("")

    def test_directory_path_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ImageInputError, "neither an existing file"):
                load_image(tmp)

    def test_non_http_schemes_raise(self):
        for value in ("ftp://example.com/a.png", "file:///tmp/a.png"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ImageInputError, "neither an existing file"):
                    load_image(value)

if __name__ == "__main__":
    unittest.main()
