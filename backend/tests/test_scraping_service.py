import pytest

from app.services.scraping_service import ScrapingError, extract_visible_text, sanitize_url, scrape_url


def test_sanitize_url_accepts_http_and_https():
    assert sanitize_url("https://example.com/docs") == "https://example.com/docs"
    assert sanitize_url("http://example.com") == "http://example.com"


def test_sanitize_url_rejects_invalid_protocols():
    with pytest.raises(ScrapingError):
        sanitize_url("ftp://example.com/file.txt")

    with pytest.raises(ScrapingError):
        sanitize_url("example.com/no-scheme")


def test_extract_visible_text_removes_scripts_and_styles():
    html = """
    <html>
      <head>
        <style>body { display:none; }</style>
        <script>console.log("hidden")</script>
      </head>
      <body>
        <h1>Chapter 1</h1>
        <p>Visible content for learning.</p>
      </body>
    </html>
    """
    text = extract_visible_text(html)
    assert "Visible content for learning." in text
    assert "console.log" not in text
    assert "display:none" not in text


def test_scrape_url_extracts_html_title_and_text(monkeypatch):
    import asyncio
    class FakeResponse:
        def __init__(self):
            self.headers = {"content-type": "text/html"}
            self.text = "<html><head><title>Lesson</title></head><body><p>Energy is conserved.</p></body></html>"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr("app.services.scraping_service.httpx.AsyncClient", FakeClient)
    result = asyncio.run(scrape_url("https://example.com/lesson"))

    assert result["title"] == "Lesson"
    assert "Energy is conserved." in result["text"]
    assert result["sourceUrl"] == "https://example.com/lesson"
    assert result["sourceFormat"] == "url"
