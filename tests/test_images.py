"""Unit tests for image input parsing."""

import base64
import tempfile
import unittest
import urllib.error
import urllib.request
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

    def test_rejects_garbage_payload(self):
        with self.assertRaises(ImageInputError):
            load_image("data:image/png;base64,@@@not-base64@@@")

    def test_rejects_oversize_payload(self):
        with mock.patch.object(images, "MAX_IMAGE_BYTES", 1024):
            big = base64.b64encode(b"x" * 2048).decode("ascii")
            with self.assertRaisesRegex(ImageInputError, "exceeds 1024 byte limit"):
                load_image(f"data:image/png;base64,{big}")


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

    def test_redirect_to_ftp_is_blocked(self):
        handler = _HttpOnlyRedirectHandler()
        req = urllib.request.Request("https://example.com/a.png")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            handler.redirect_request(req, None, 302, "Found", {}, "ftp://internal/steal.png")
        if ctx.exception.fp:
            ctx.exception.fp.close()

    def test_redirect_to_http_is_followed(self):
        handler = _HttpOnlyRedirectHandler()
        req = urllib.request.Request("https://example.com/a.png")
        # The handler delegates to the stdlib, which raises for malformed
        # targets with no scheme; a plain http target must not raise here.
        try:
            handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.com/b.png")
        except urllib.error.HTTPError:
            self.fail("https redirect should be delegated, not blocked")


class InvalidInputTests(unittest.TestCase):
    def test_nonexistent_path_raises(self):
        with self.assertRaises(ImageInputError):
            load_image("/definitely/not/a/real/file.png")


if __name__ == "__main__":
    unittest.main()
