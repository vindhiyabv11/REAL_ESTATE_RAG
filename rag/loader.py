"""Fetch a URL and extract ONLY the main article content (no nav/header/footer/ads)."""
import re

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RealEstateRAG/1.0; personal research use) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside",
               "form", "iframe", "svg", "button", "figure", "figcaption"]
_NOISE_TOKENS = {"ad", "ads", "advert", "advertisement", "promo", "newsletter", "related",
                 "share", "social", "comments", "comment", "footer", "header", "nav",
                 "navbar", "subscribe", "sidebar", "breadcrumb", "banner", "cookie",
                 "popup", "modal", "recommended", "trending"}


class LoaderError(Exception):
    """Raised when a page cannot be fetched or has no usable article text."""


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _fallback_extract(html: str) -> tuple[str, str | None]:
    """BeautifulSoup extraction used when trafilatura is missing or returns too little."""
    soup = BeautifulSoup(html, "lxml")
    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    for el in list(root.find_all(True)):
        if getattr(el, "decomposed", False) or el.attrs is None:
            continue
        cls = el.get("class") or []
        tokens = set(re.split(r"[\s_-]+", " ".join(cls) + " " + str(el.get("id") or "")))
        if tokens & _NOISE_TOKENS:
            el.decompose()

    parts = [el.get_text(" ", strip=True) for el in root.find_all(["h2", "h3", "p", "li"])]
    return "\n\n".join(p for p in parts if p), title


def extract_main_text(html: str, url: str = "") -> tuple[str, str | None, str | None]:
    """Return (text, title, published_date) for the main article body."""
    text, title, date = "", None, None
    try:
        import trafilatura

        text = trafilatura.extract(
            html, url=url or None, include_comments=False, include_tables=False,
            include_images=False, include_links=False, favor_precision=True,
        ) or ""
        meta = trafilatura.extract_metadata(html, default_url=url or None, extensive = False)
        if meta is not None:
            title, date = meta.title, meta.date
    except Exception:
        text = ""

    if len(text) < config.MIN_ARTICLE_CHARS:
        fb_text, fb_title = _fallback_extract(html)
        if len(fb_text) > len(text):
            text = fb_text
        title = title or fb_title
    return text, title, date


def load_url(url: str, fallback: dict | None = None) -> Document:
    """Fetch `url` and return a Document holding the article text and metadata."""
    fallback = fallback or {}
    try:
        html = fetch_html(url)
    except requests.RequestException as exc:
        raise LoaderError(f"download failed ({exc.__class__.__name__}: {exc})") from exc

    text, title, date = extract_main_text(html, url)
    if not text.strip():
        raise LoaderError("no article text found (page may be paywalled or JavaScript-rendered)")

    return Document(
        page_content=text,
        metadata={
            "url": url,
            "title": (fallback.get("title") or title or url).strip(),
            "published": str(fallback.get("published") or date or "")[:10],
            "source": fallback.get("source") or re.sub(r"^https?://(www\.)?", "", url).split("/")[0],
        },
    )

    