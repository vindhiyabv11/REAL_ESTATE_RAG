from rag.loader import _fallback_extract, extract_main_text

HTML = """
<html><head><title>Site | Fed cuts rates</title></head><body>
<header><nav>Home | News | Contact</nav></header>
<article>
  <h1>Fed cuts rates again</h1>
  <div class="newsletter-signup">Subscribe to our newsletter</div>
  <p>The Federal Reserve cut its benchmark interest rate by a quarter point on Wednesday, bringing the range to 4.25% to 4.50%.</p>
  <p>Mortgage rates, however, rose after the announcement because investors expect fewer cuts in 2025, pushing Treasury yields higher.</p>
  <p>Economists said borrowing costs for homebuyers are unlikely to fall meaningfully until inflation cools further and the labor market softens.</p>
  <div class="related-articles"><p>Related: Ten ways to save on a mortgage</p></div>
</article>
<footer>All rights reserved. Privacy policy</footer>
</body></html>
"""


def test_fallback_extracts_article_only():
    text, title = _fallback_extract(HTML)
    assert "4.25% to 4.50%" in text
    assert "Home | News | Contact" not in text
    assert "All rights reserved" not in text
    assert "Subscribe to our newsletter" not in text
    assert "Ten ways to save" not in text
    assert title == "Fed cuts rates again"


def test_extract_main_text_end_to_end():
    text, _title, _date = extract_main_text(HTML, "https://example.com/fed")
    assert "Treasury yields" in text
    assert "Home | News | Contact" not in text
    assert "All rights reserved" not in text

    