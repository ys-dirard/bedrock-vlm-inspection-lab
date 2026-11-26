from summarize_token_check import claude_resized, doc_expected
from token_check import NO_TEMPERATURE, PRICES


def test_claude_resized_matches_doc_examples():
    # Anthropic の vision-coordinates の例：1075x1520 → 924x1307、1920x1080 → 1456x819
    assert claude_resized(1075, 1520) == (924, 1307)
    assert claude_resized(1920, 1080) == (1456, 819)
    assert claude_resized(768, 585) == (768, 585)


def test_doc_expected():
    assert doc_expected("jp.amazon.nova-2-lite-v1:0", 1404, 1070)[0] == "230"
    assert doc_expected("jp.anthropic.claude-sonnet-4-6", 1404, 1070)[0] == "1530"
    assert doc_expected("google.gemma-3-4b-it", 768, 585)[0] == "256"
    assert doc_expected("qwen.qwen3-vl-235b-a22b", 768, 585)[0] == ""


def test_prices_and_no_temperature():
    assert NO_TEMPERATURE <= set(PRICES)
    assert all(i >= 0 and o >= 0 for i, o in PRICES.values())
