import pytest

from token_check import Usage, claude_visual_tokens, cost_usd, resized_size


def test_cost_usd():
    # 1,000 入力 × 1.10 + 50 出力 × 5.50（USD / 100万トークン）
    assert cost_usd(Usage(1_000, 50), 1.10, 5.50) == pytest.approx(0.001375)
    assert cost_usd(Usage(0, 0), 1.0, 1.0) == 0.0


def test_cost_usd_rejects_negative():
    with pytest.raises(ValueError):
        cost_usd(Usage(-1, 0), 1.0, 1.0)
    with pytest.raises(ValueError):
        cost_usd(Usage(1, 0), -1.0, 1.0)


def test_resized_size():
    assert resized_size(1404, 1070, 768) == (768, 585)
    assert resized_size(1070, 1404, 768) == (585, 768)
    assert resized_size(500, 400, 768) == (500, 400)
    with pytest.raises(ValueError):
        resized_size(0, 10, 768)


def test_claude_visual_tokens_matches_doc_examples():
    # Anthropic の Vision のドキュメントの表：1000x1000 → 1,296、200x200 → 64
    assert claude_visual_tokens(1000, 1000) == 1296
    assert claude_visual_tokens(200, 200) == 64
