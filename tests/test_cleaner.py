from rag.cleaner import clean_text


def test_removes_boilerplate_lines():
    raw = "Subscribe to CNBC PRO\nThe Fed cut rates by 0.25 percentage point on Wednesday.\nAdvertisement\n© 2024 CNBC LLC. All Rights Reserved."
    out = clean_text(raw)
    assert "Subscribe" not in out
    assert "Advertisement" not in out
    assert "All Rights Reserved" not in out
    assert "The Fed cut rates by 0.25 percentage point on Wednesday." in out


def test_keeps_real_sentences_that_start_with_boilerplate_words():
    raw = "Shares of Zillow rose 3% after the report.\nFollowing the decision, mortgage rates climbed."
    out = clean_text(raw)
    assert "Shares of Zillow rose 3% after the report." in out
    assert "Following the decision, mortgage rates climbed." in out


def test_drops_duplicate_lines_and_short_nav_fragments():
    raw = "Home\nMortgage rates hit 7% this week, according to Freddie Mac.\nMortgage rates hit 7% this week, according to Freddie Mac."
    out = clean_text(raw)
    assert out.count("Mortgage rates hit 7%") == 1
    assert "Home" not in out.splitlines()


def test_normalizes_whitespace():
    out = clean_text("Rates\xa0are   up.\n\n\n\nHomes sell fast in this market.")
    assert "  " not in out
    assert "\n\n\n" not in out


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""