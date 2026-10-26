"""Check the direct sample-download link and the exact bytes it serves."""

from base64 import b64decode
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from downloads import sample_download_url


class SampleLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.samples = []
        self.previews = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a" and "sample-download" in attributes.get("class", "").split():
            self.samples.append(attributes)
        if tag == "a" and "sample-preview" in attributes.get("class", "").split():
            self.previews.append(attributes)


def sample_links(app):
    parser = SampleLinkParser()
    for element in app.markdown:
        parser.feed(element.value)
    return parser


def sample_download_href(app):
    return sample_links(app).samples[0]["href"]


def assert_sample_download(test, app, static_root, expected_bytes):
    """Check the embedded download and the saved CSV preview independently."""
    parser = sample_links(app)
    test.assertEqual(len(parser.samples), 1)
    anchor = parser.samples[0]
    test.assertEqual(anchor["download"], "test_data.csv")
    test.assertTrue(anchor["href"].startswith("data:text/csv;base64,"))
    decoded = b64decode(anchor["href"].split(",", 1)[1], validate=True)
    test.assertEqual(decoded, expected_bytes)
    test.assertEqual(len(parser.previews), 1)
    preview = parser.previews[0]
    test.assertTrue(preview["href"].startswith("app/static/downloads/"))
    relative = preview["href"].removeprefix("app/static/")
    destination = Path(static_root) / relative
    test.assertEqual(destination.name, "test_data.csv")
    test.assertEqual(destination.read_bytes(), expected_bytes)
    return decoded


class SampleDownloadTests(unittest.TestCase):
    def test_sample_urls_are_stable_and_different_dataset_bytes_stay_separate(self):
        first = b"feature,target\n10,normal\n20,attack\n"
        second = b"feature,target\n30,red\n40,blue\n"
        with TemporaryDirectory(prefix="sample-download-test-") as static_root:
            first_url = sample_download_url(first, static_root=static_root)
            second_url = sample_download_url(second, static_root=static_root)
            repeated_url = sample_download_url(first, static_root=static_root)
            self.assertEqual(first_url, repeated_url)
            self.assertNotEqual(first_url, second_url)
            for url, expected in ((first_url, first), (second_url, second)):
                self.assertTrue(url.startswith("app/static/downloads/"))
                destination = Path(static_root) / url.removeprefix("app/static/")
                self.assertEqual(destination.name, "test_data.csv")
                self.assertEqual(destination.read_bytes(), expected)
            self.assertEqual(len(list(Path(static_root).rglob("test_data.csv"))), 2)


if __name__ == "__main__":
    unittest.main()
