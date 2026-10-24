"""Second-pass cleaning of extracted article text."""
import re

_BOILERPLATE = [
    r"^(subscribe|sign up|sign in|log in|follow us|follow @)\b.{0,100}$",
    r"^(read more|read also|read next|related|related articles?|more from|see also|watch|listen|advertisement|sponsored|click here)\b.{0,80}$",
    r"^(share|tweet|email|print|save|copy link|comments?)(\s+(this|article|story))?:?$",
    r"^(©|copyright)\b.*$",
    r".*\ball rights reserved\b.*",
    r"^(terms of (use|service)|privacy policy|cookie (policy|settings)|contact us|about us)\b.{0,40}$",
    r"^(photo|image|getty images|credit|source)\s*[:|-].{0,80}$",
    r"^https?://\S+$",
    r"^(key points|advertisement|skip to (main )?content|menu|search)$",
]
_BOILERPLATE_RE = [re.compile(p, re.IGNORECASE) for p in _BOILERPLATE]
_TERMINAL_PUNCT = (".", "!", "?", '"', "”", "’", ":", ")")


def _is_boilerplate(line: str) -> bool:
    if any(rx.match(line) for rx in _BOILERPLATE_RE):
        return True
    # Very short fragments without punctuation are usually nav labels / tags.
    return len(line.split()) <= 2 and not line.endswith(_TERMINAL_PUNCT)
    


def clean_text(text: str) -> str:
    """Return article text with boilerplate, duplicate lines and extra whitespace removed."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)

    seen = set()
    kept = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            kept.append("")
            continue
        if _is_boilerplate(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


    