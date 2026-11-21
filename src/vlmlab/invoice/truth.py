"""架空の適格請求書の正解（JSON）を作る（docs/contracts.md の CI-IT1〜IT5）。

会社名・住所・品目はすべて造語か一般名詞。実在の事業者を指さない。
登録番号は国税庁の検査用数字の計算方法（https://www.houjin-bangou.nta.go.jp/documents/checkdigit.pdf）に合う架空の番号。
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any

SEED = 42
N_INVOICES = 60
N_ISSUERS = 30

# 発行元（造語＋業種）。実在の会社名と重ならないよう、意味の無いカタカナの造語にしている
ISSUER_STEMS = [
    "ミナホル", "クレスミナ", "トワニカ", "ハルメリ", "ソラネル", "キリフネ", "ユノカタ", "モリサエ",
    "アオトビ", "ナギヘル", "コトミラ", "ヒバリノ", "スズカネ", "ワカナト", "フミヨル", "カゼモト",
    "ツキホナ", "ミズラギ", "サクヤネ", "ホシゴロ", "ルリカゼ", "ノゾミル", "タマユラ", "イロハネ",
    "マドカリ", "シオサイノ", "ヨツバネ", "カナタリ", "ヒナセル", "ミオツク",
]
ISSUER_KINDS = ["商事", "工業", "精機", "物産", "印刷", "食品", "電機", "化成", "設計", "運輸"]
RECIPIENT_STEMS = [
    "アサギリ", "ユメカド", "ホノカミ", "セツナル", "リンドウネ", "カスミネ", "トモシビノ", "ハヤテル",
    "ミツハシル", "コハクネ", "シノノメル", "アカネリ", "ヨイマチ", "サザナミル", "ウタカネ", "ナナホシル",
]
# 架空の地名（実在の市区町村名と重ならない造語）
CITIES = ["東方市", "緑ヶ丘町", "新港市", "北原村", "桜台市", "若葉町", "西浜市", "朝凪町"]
PREFS = ["東京都", "神奈川県", "大阪府", "愛知県", "福岡県", "北海道"]

ITEMS_STD = [
    "コピー用紙 A4", "ボールペン 黒", "クリアファイル", "トナーカートリッジ", "USBメモリ 32GB",
    "LANケーブル 3m", "段ボール箱 大", "梱包テープ", "マウスパッド", "ホワイトボードマーカー",
    "保守サービス料", "設置作業費", "配送料", "清掃用洗剤", "ゴム手袋", "電池 単3形",
    "ファイルボックス", "延長コード 5m", "名刺印刷", "封筒 長形3号", "付箋 75mm", "ハサミ",
    "ノートパソコンスタンド", "キーボード", "作業用マスク", "結束バンド", "ラベルシール", "電卓",
    "シュレッダー用袋", "掲示用パネル",
]
ITEMS_REDUCED = [
    "緑茶 500ml", "ミネラルウォーター 2L", "コーヒー豆 200g", "せんべい詰合せ", "弁当", "おにぎり",
    "缶コーヒー", "クッキー", "カップスープ", "オレンジジュース 1L", "牛乳 1L", "パン詰合せ",
    "ほうじ茶 ティーバッグ", "チョコレート", "ナッツ 100g", "バナナ", "りんご", "ゼリー飲料",
    "紅茶 ティーバッグ", "麦茶 2L", "あめ", "ヨーグルト", "米 5kg", "みそ汁 即席",
    "カップ麺", "炭酸水 500ml", "ドライフルーツ", "はちみつ", "梅干し", "のり",
]

FIELDS = [
    "invoice_number", "issue_date", "issuer_name", "registration_number", "recipient_name",
    "subtotal_10", "tax_10", "subtotal_8", "tax_8", "total",
]
MONEY_FIELDS = ["subtotal_10", "tax_10", "subtotal_8", "tax_8", "total"]


def check_digit(base12: str) -> str:
    """法人番号の検査用数字（CI-IT1）。右端を1桁目とし、9 −（奇数桁の和 + 偶数桁の和×2）を 9 で割った余り。"""
    if len(base12) != 12 or not base12.isascii() or not base12.isdigit():
        raise ValueError(f"base must be 12 ASCII digits, got {base12!r}")
    d = [int(c) for c in reversed(base12)]
    odd = sum(d[0::2])
    even = sum(d[1::2])
    return str(9 - (odd + even * 2) % 9)


def is_valid_registration_number(rn: Any) -> bool:
    """T＋13桁で、先頭の数字が残り12桁の検査用数字に合うか（CI-IT1）。"""
    return (isinstance(rn, str) and len(rn) == 14 and rn[0] == "T" and rn[1:].isascii()
            and rn[1:].isdigit() and check_digit(rn[2:]) == rn[1])


def registration_number(rng: random.Random) -> str:
    base = "".join(rng.choice("0123456789") for _ in range(12))
    return "T" + check_digit(base) + base


def make_issuers(rng: random.Random) -> list[dict[str, str]]:
    """発行元30社（名前・住所・登録番号）。登録番号は重複させない（CI-IT2）。"""
    issuers, seen = [], set()
    for i, stem in enumerate(ISSUER_STEMS[:N_ISSUERS]):
        kind = ISSUER_KINDS[i % len(ISSUER_KINDS)]
        name = f"株式会社{stem}{kind}" if i % 2 == 0 else f"{stem}{kind}株式会社"
        rn = registration_number(rng)
        while rn in seen:
            rn = registration_number(rng)
        seen.add(rn)
        addr = f"{rng.choice(PREFS)}{rng.choice(CITIES)}{rng.randint(1, 9)}丁目{rng.randint(1, 30)}番{rng.randint(1, 20)}号"
        issuers.append({"name": name, "address": addr, "registration_number": rn})
    return issuers


def make_recipients() -> list[str]:
    kinds = ["株式会社", "合同会社", "有限会社"]
    return [f"{kinds[i % 3]}{stem}" for i, stem in enumerate(RECIPIENT_STEMS)]


def make_truth(rng: random.Random, i: int, issuers: list[dict[str, str]], recipients: list[str]) -> dict[str, Any]:
    """1枚ぶんの正解（CI-IT3、IT4）。税は税率ごとに1回、端数を切り捨てる。"""
    items = []
    n = rng.randint(3, 8)
    n_reduced = rng.randint(0, min(3, n))
    flags = [True] * n_reduced + [False] * (n - n_reduced)
    rng.shuffle(flags)
    for reduced in flags:
        qty = rng.randint(1, 20)
        price = rng.choice(range(100, 50001, 10))
        items.append({
            "description": rng.choice(ITEMS_REDUCED if reduced else ITEMS_STD),
            "quantity": qty, "unit_price": price, "amount": qty * price, "reduced_rate": reduced,
        })
    s10 = sum(x["amount"] for x in items if not x["reduced_rate"])
    s8 = sum(x["amount"] for x in items if x["reduced_rate"])
    t10, t8 = s10 * 10 // 100, s8 * 8 // 100
    issue = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 300))
    issuer = rng.choice(issuers)
    return {
        "invoice_id": f"inv{i:04d}",
        "invoice_number": f"INV-2026{i:04d}",
        "issue_date": issue.isoformat(),
        "date_style": "wareki" if i % 2 == 0 else "seireki",  # 半数を「令和8年…」で書く（正解は ISO で同じ）
        "issuer_name": issuer["name"],
        "issuer_address": issuer["address"],
        "registration_number": issuer["registration_number"],
        "recipient_name": rng.choice(recipients),
        "subtotal_10": s10, "tax_10": t10, "subtotal_8": s8, "tax_8": t8,
        "total": s10 + t10 + s8 + t8,
        "line_items": items,
    }


def make_all(seed: int = SEED, n: int = N_INVOICES) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """正解 n 枚と発行元の一覧。同じ seed なら同じ結果（CI-IT5）。"""
    rng = random.Random(seed)
    issuers = make_issuers(rng)
    recipients = make_recipients()
    truths = [make_truth(rng, i, issuers, recipients) for i in range(1, n + 1)]
    return truths, issuers


def consistent(p: dict[str, Any] | None) -> dict[str, bool]:
    """読み取った値どうしの整合（正解は見ない。CI-IS6）。

    - registration_number: T＋13桁で検査用数字が合う
    - sum: 小計10 + 税10 + 小計8 + 税8 = 合計
    - tax: 税10 = 小計10×10÷100（切り捨て）、税8 = 小計8×8÷100（切り捨て）
    金額が整数（bool は除く）でなければ sum・tax は False。
    """
    p = p or {}
    rn_ok = is_valid_registration_number(p.get("registration_number"))
    vals = [p.get(k) for k in MONEY_FIELDS]
    ints = all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
    if ints:
        s10, t10, s8, t8, tot = vals
        sum_ok = s10 + t10 + s8 + t8 == tot
        tax_ok = t10 == s10 * 10 // 100 and t8 == s8 * 8 // 100
    else:
        sum_ok = tax_ok = False
    return {"rn_ok": rn_ok, "sum_ok": sum_ok, "tax_ok": tax_ok, "all_ok": rn_ok and sum_ok and tax_ok}
