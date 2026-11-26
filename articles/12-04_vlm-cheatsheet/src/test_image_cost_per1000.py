import pytest

from image_cost_per1000 import build, image_cost_per1000


def test_image_cost_per1000():
    # 1,534 トークン × 3.3 USD / 100万 × 1,000枚 = 5.0622 USD
    assert image_cost_per1000(1534, 3.3) == pytest.approx(5.0622)
    assert image_cost_per1000(0, 3.3) == 0
    with pytest.raises(ValueError):
        image_cost_per1000(-1, 3.3)


def test_build_skips_failed_and_uses_call_price():
    rows = [
        {"model_id": "jp.amazon.nova-2-lite-v1:0", "variant": "original", "image_tokens_measured": "232"},
        {"model_id": "jp.amazon.nova-2-lite-v1:0", "variant": "long768", "image_tokens_measured": "232"},
        {"model_id": "global.xai.grok-4.6", "variant": "(失敗)", "image_tokens_measured": ""},
    ]
    out = build(rows)
    assert len(out) == 1
    assert out[0]["input_usd_per_mtok"] == 0.396
    assert out[0]["long768_over_original"] == 1.0
