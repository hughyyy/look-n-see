"""Unit tests for the shared response-sanity classifier."""

import unittest

from look_n_see.sanity import annotate, classify, severe_flags


class ClassifyTests(unittest.TestCase):
    def test_refusal_anchored_to_sentence_start(self):
        self.assertIn(
            "refusal", classify("I am a text-based AI and cannot fulfill this")
        )
        self.assertIn("refusal", classify("Sorry. I cannot process images."))
        self.assertIn("refusal", classify("I cannot"))

    def test_ordinary_unable_to_phrase_is_not_a_refusal(self):
        text = (
            "The dialog is unable to load its contents because the network is "
            "offline, and the small label reads 'error'."
        )
        self.assertEqual(classify(text), ["normal"])
        self.assertEqual(severe_flags(text), [])

    def test_gen_payload_and_json_markers(self):
        self.assertIn("gen_payload", classify('{"model": "dalle.text2im"}'))
        self.assertIn("json_markers", classify('{"point": [1, 2], "label": "x"}'))

    def test_empty_and_terse(self):
        self.assertEqual(classify(""), ["empty"])
        self.assertEqual(classify("   \n"), ["empty"])
        self.assertEqual(classify("NONE"), ["terse"])
        self.assertEqual(classify("x" * 60), ["normal"])

    def test_severe_flags_excludes_quality_signals(self):
        self.assertEqual(severe_flags("NONE"), [])
        self.assertEqual(severe_flags(""), ["empty"])
        self.assertIn("refusal", severe_flags("I cannot fulfill this request."))


class AnnotateTests(unittest.TestCase):
    def test_normal_text_unchanged(self):
        text = "A dark mask over a pane with a Wait message chip, visible now."
        self.assertEqual(annotate(text), text)

    def test_refusal_gets_warning(self):
        annotated = annotate("I cannot fulfill this request.")
        self.assertIn("I cannot fulfill", annotated)
        self.assertIn("[look-n-see warning: response flagged as refusal", annotated)

    def test_empty_returns_warning_only(self):
        annotated = annotate("")
        self.assertTrue(annotated.startswith("[look-n-see warning"))
        self.assertIn("empty", annotated)


if __name__ == "__main__":
    unittest.main()
