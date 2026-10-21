"""Unit tests for image input parsing."""

import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from look_n_see.images import ImageInputError, load_image, to_data_uri

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


class FilePathTests(unittest.TestCase):
    def test_loads_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.png"
            path.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
            media, data = load_image(str(path))
            self.assertEqual(media, "image/png")
            self.assertEqual(data, bytes.fromhex("89504e470d0a1a0a"))


class UrlTests(unittest.TestCase):
    def test_fetches_and_uses_content_type(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.headers.get_content_type.return_value = "image/jpeg"
        fake.read.return_value = b"\xff\xd8\xff\xe0"
        with mock.patch("urllib.request.urlopen", return_value=fake) as urlopen:
            media, data = load_image("https://example.com/shot.jpg")
        urlopen.assert_called_once_with("https://example.com/shot.jpg", timeout=30.0)
        self.assertEqual(media, "image/jpeg")
        self.assertEqual(data, b"\xff\xd8\xff\xe0")

    def test_fetch_failure_raises(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
            with self.assertRaises(ImageInputError):
                load_image("https://example.com/missing.png")


class InvalidInputTests(unittest.TestCase):
    def test_nonexistent_path_raises(self):
        with self.assertRaises(ImageInputError):
            load_image("/definitely/not/a/real/file.png")


if __name__ == "__main__":
    unittest.main()
