# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for document crawler — link extraction and recursive fetching."""

import pytest

from constat.discovery.doc_tools._crawler import extract_links, crawl_document, _normalize_url
from constat.discovery.doc_tools._transport import FetchResult


class TestExtractLinks:
    def test_html_href(self):
        html = '<a href="https://example.com/page1">Link1</a> <a href="/page2">Link2</a>'
        links = extract_links(html, "html", "https://example.com/")
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links

    def test_markdown_links(self):
        md = "See [docs](https://example.com/docs) and [faq](/faq)"
        links = extract_links(md, "markdown", "https://example.com/")
        assert "https://example.com/docs" in links
        assert "https://example.com/faq" in links

    def test_skips_anchors_and_mailto(self):
        html = '<a href="#section">Anchor</a> <a href="mailto:a@b.com">Email</a>'
        links = extract_links(html, "html", "https://example.com/")
        assert links == []

    def test_text_returns_empty(self):
        links = extract_links("some plain text with https://example.com", "text", None)
        assert links == []

    def test_relative_resolution(self):
        html = '<a href="sub/page.html">Sub</a>'
        links = extract_links(html, "html", "https://example.com/docs/")
        assert links == ["https://example.com/docs/sub/page.html"]


class _FakeConfig:
    def __init__(self, **kwargs):
        self.url = kwargs.get("url")
        self.content = kwargs.get("content")
        self.path = kwargs.get("path")
        self.headers = kwargs.get("headers", {})
        self.follow_links = kwargs.get("follow_links", True)
        self.max_depth = kwargs.get("max_depth", 2)
        self.max_documents = kwargs.get("max_documents", 20)
        self.link_pattern = kwargs.get("link_pattern")
        self.same_domain_only = kwargs.get("same_domain_only", True)
        self.username = None
        self.password = None
        self.port = None
        self.key_path = None
        self.aws_profile = None
        self.aws_region = None


class TestCrawlDocument:
    def test_single_page_no_links(self):
        """Root page with no links returns just the root."""

        def fake_fetch(config, config_dir):
            return FetchResult(data=b"Hello world", detected_mime="text/plain")

        config = _FakeConfig(url="https://example.com/page.txt")
        results = crawl_document(config, None, fake_fetch)
        assert len(results) == 1
        assert results[0][0] == "https://example.com/page.txt"

    def test_follows_links(self):
        """Root HTML page links to a child; both are returned."""
        pages = {
            "https://example.com/": FetchResult(
                data=b'<a href="/child">Link</a>',
                detected_mime="text/html",
            ),
            "https://example.com/child": FetchResult(
                data=b"Child content",
                detected_mime="text/plain",
            ),
        }

        def fake_fetch(config, config_dir):
            url = config.url
            if url in pages:
                return pages[url]
            raise ValueError(f"Unknown URL: {url}")

        config = _FakeConfig(url="https://example.com/")
        results = crawl_document(config, None, fake_fetch)
        urls = [url for url, _ in results]
        assert "https://example.com/" in urls
        assert "https://example.com/child" in urls

    def test_respects_max_documents(self):
        """Stops after max_documents are fetched."""

        def fake_fetch(config, config_dir):
            url = config.url
            return FetchResult(
                data=f'<a href="{url}1">Link</a>'.encode(),
                detected_mime="text/html",
            )

        config = _FakeConfig(url="https://example.com/", max_documents=3)
        results = crawl_document(config, None, fake_fetch)
        assert len(results) <= 3

    def test_same_domain_only(self):
        """Skips links to different domains."""
        pages = {
            "https://example.com/": FetchResult(
                data=b'<a href="https://other.com/page">External</a>',
                detected_mime="text/html",
            ),
        }

        def fake_fetch(config, config_dir):
            return pages.get(config.url, FetchResult(data=b"", detected_mime="text/plain"))

        config = _FakeConfig(url="https://example.com/", same_domain_only=True)
        results = crawl_document(config, None, fake_fetch)
        urls = [url for url, _ in results]
        assert "https://other.com/page" not in urls

    def test_link_pattern_filter(self):
        """Only follows links matching the pattern."""
        pages = {
            "https://example.com/": FetchResult(
                data=b'<a href="/docs/api">API</a><a href="/blog/post">Blog</a>',
                detected_mime="text/html",
            ),
            "https://example.com/docs/api": FetchResult(
                data=b"API docs", detected_mime="text/plain"
            ),
        }

        def fake_fetch(config, config_dir):
            return pages.get(config.url, FetchResult(data=b"", detected_mime="text/plain"))

        config = _FakeConfig(url="https://example.com/", link_pattern=r"/docs/")
        results = crawl_document(config, None, fake_fetch)
        urls = [url for url, _ in results]
        assert "https://example.com/docs/api" in urls
        assert "https://example.com/blog/post" not in urls

    def test_deduplication(self):
        """Same URL is not fetched twice."""

        call_count = {}

        def fake_fetch(config, config_dir):
            call_count[config.url] = call_count.get(config.url, 0) + 1
            if config.url == "https://example.com/":
                return FetchResult(
                    data=b'<a href="/page">A</a><a href="/page">B</a>',
                    detected_mime="text/html",
                )
            return FetchResult(data=b"page", detected_mime="text/plain")

        config = _FakeConfig(url="https://example.com/")
        results = crawl_document(config, None, fake_fetch)
        # /page should only appear once
        page_results = [url for url, _ in results if "page" in url]
        assert len(page_results) == 1


class TestNormalizeUrl:
    def test_strips_fragment(self):
        assert _normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_strips_trailing_slash(self):
        assert _normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_preserves_root(self):
        assert _normalize_url("https://example.com/") == "https://example.com/"
