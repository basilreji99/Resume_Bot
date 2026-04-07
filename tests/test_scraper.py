import pytest
from unittest.mock import patch
from app.scraper import scrape_url, clean_pasted_text, is_blocked, get_domain


# ── Domain helpers ────────────────────────────────────────────
def test_get_domain_strips_www():
    assert get_domain("https://www.linkedin.com/in/basil") == "linkedin.com"


def test_get_domain_no_www():
    assert get_domain("https://github.com/basil") == "github.com"


def test_is_blocked_linkedin():
    assert is_blocked("https://www.linkedin.com/in/basil") is True


def test_is_blocked_github():
    assert is_blocked("https://github.com/basil") is False


# ── scrape_url ────────────────────────────────────────────────
def test_scrape_url_blocked_domain():
    """LinkedIn URLs should raise ValueError before making any network call."""
    with pytest.raises(ValueError, match="linkedin.com blocks"):
        scrape_url("https://www.linkedin.com/in/basil")


def test_scrape_url_invalid_scheme():
    """Non-http URLs should be rejected immediately."""
    with pytest.raises(ValueError, match="valid URL"):
        scrape_url("ftp://example.com/page")


def test_scrape_url_fetch_failure():
    """
    If trafilatura.fetch_url returns None (unreachable site),
    we should get a helpful ValueError, not a crash.

    patch() temporarily replaces trafilatura.fetch_url with a
    fake that returns None — no real network call is made.
    """
    with patch("app.scraper.trafilatura.fetch_url", return_value=None):
        with pytest.raises(ValueError, match="Could not reach"):
            scrape_url("https://example.com")


def test_scrape_url_empty_content():
    """
    If the page loads but trafilatura extracts nothing (JS-heavy page),
    we should get a helpful ValueError.
    """
    with patch("app.scraper.trafilatura.fetch_url", return_value="<html></html>"):
        with patch("app.scraper.trafilatura.extract", return_value=""):
            with pytest.raises(ValueError, match="Could not extract"):
                scrape_url("https://example.com")


def test_scrape_url_success():
    """Happy path: fetch returns HTML, extract returns real text.
    The fake text must be > 50 chars to pass scraper.py's length check."""
    fake_html = "<html><body><p>Basil Ahmed is a software engineer.</p></body></html>"
    fake_text = (
        "Basil Ahmed is a software engineer with 5 years of experience "
        "in Python, machine learning, and cloud infrastructure."
    )
    with patch("app.scraper.trafilatura.fetch_url", return_value=fake_html):
        with patch("app.scraper.trafilatura.extract", return_value=fake_text):
            result = scrape_url("https://example.com")
            assert "Basil Ahmed" in result


# ── clean_pasted_text ─────────────────────────────────────────
def test_clean_pasted_text_too_short():
    with pytest.raises(ValueError, match="too short"):
        clean_pasted_text("Hi")


def test_clean_pasted_text_removes_extra_blank_lines():
    messy = "Line one\n\n\n\nLine two\n\n\n\nLine three"
    result = clean_pasted_text(messy)
    # Should not have more than one consecutive blank line
    assert "\n\n\n" not in result


def test_clean_pasted_text_preserves_content():
    text = "Basil Ahmed\nSoftware Engineer\n5 years of Python experience."
    result = clean_pasted_text(text)
    assert "Basil Ahmed" in result
    assert "Python" in result