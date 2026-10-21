"""Unit tests for prompt assembly."""

import unittest

from look_n_see.prompts import (
    ASK_DEFAULT,
    DESCRIBE_DEFAULT,
    SCREENSHOT_DEFAULT,
    build_messages,
)

URI = "data:image/png;base64,AAAA"


class BuildMessagesTests(unittest.TestCase):
    def test_default_prompt_when_none(self):
        (msg,) = build_messages(URI, None, DESCRIBE_DEFAULT)
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"][1]["text"], DESCRIBE_DEFAULT)

    def test_default_prompt_when_blank(self):
        (msg,) = build_messages(URI, "   ", SCREENSHOT_DEFAULT)
        self.assertEqual(msg["content"][1]["text"], SCREENSHOT_DEFAULT)

    def test_custom_prompt_wins(self):
        (msg,) = build_messages(URI, "Focus on the error dialog", ASK_DEFAULT)
        self.assertEqual(msg["content"][1]["text"], "Focus on the error dialog")

    def test_message_shape(self):
        (msg,) = build_messages(URI, "q", ASK_DEFAULT)
        self.assertEqual(msg["content"][0], {"type": "image_url", "image_url": {"url": URI}})
        self.assertEqual(msg["content"][1], {"type": "text", "text": "q"})


if __name__ == "__main__":
    unittest.main()
