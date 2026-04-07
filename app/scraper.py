import trafilatura
import urllib.parse


# Sites that actively block scrapers or require login
BLOCKED_DOMAINS = ["linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com"]


def get_domain(url: str) -> str:
    """
    Extracts the domain name from a full URL.
    """
    parsed = urllib.parse.urlparse(url)
    # netloc gives us "www.linkedin.com" — we strip the "www." prefix
    domain = parsed.netloc.replace("www.", "")
    return domain


def is_blocked(url: str) -> bool:
    """
    Checks if the URL belongs to a domain we know blocks scraping.
    Returns True if blocked, False if safe to attempt.
    """
    domain = get_domain(url)
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def scrape_url(url: str) -> str:
    """
    Fetches a webpage and extracts its main readable text.

    trafilatura.fetch_url() — sends an HTTP GET request, returns raw HTML
    trafilatura.extract()   — strips HTML tags, keeps only the main content
                              (ignores navbars, footers, cookie banners etc.)

    Returns the clean text, or raises a ValueError with a user-friendly message.
    """
    # Step 1: Check for known blocked domains before even trying
    if is_blocked(url):
        domain = get_domain(url)
        raise ValueError(
            f"{domain} blocks automated access. "
            "Please copy and paste your text manually using the paste option instead."
        )

    # Step 2: Validate that the URL at least looks like a URL
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme in ("http", "https"):
        raise ValueError("Please enter a valid URL starting with http:// or https://")

    # Step 3: Fetch the raw HTML from the URL
    # This is the actual HTTP GET request — it may take a second or two
    html = trafilatura.fetch_url(url)

    if html is None:
        raise ValueError(
            "Could not reach that URL. "
            "The site may be blocking access, or the URL may be incorrect."
        )

    # Step 4: Extract clean text from the raw HTML
    # include_comments=False  — skip comment sections
    # include_tables=True     — keep table data (useful for structured profiles)
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False       # if main extraction fails, try a fallback method
    )

    if not text or len(text.strip()) < 50:
        raise ValueError(
            "Could not extract readable content from that URL. "
            "The page may require a login or be mostly images/JavaScript."
        )

    return text.strip()


def clean_pasted_text(text: str) -> str:
    """
    Cleans up raw pasted text.
    Removes excessive blank lines and leading/trailing whitespace.
    This is simple but important — messy input leads to messy chunks later.
    """
    if not text or len(text.strip()) < 30:
        raise ValueError(
            "The pasted text is too short. "
            "Please paste a more complete profile or document."
        )

    # Split into lines, strip each, remove runs of more than one blank line
    lines = text.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(stripped)
            prev_blank = False

    return "\n".join(cleaned).strip()