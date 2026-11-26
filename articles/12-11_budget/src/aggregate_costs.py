"""各記事の費用の記録（results/cost_total.md）を記事ごと・モデルごとに集計する（12/11 の記事）。

目的: 記事の【結果：12/10 までの累計】と内訳を、各記事の記録だけから出す。
- 入力: 各記事の cost_total.md の表の値を、下の ROWS に写したもの（写し元のパスは source 列）
- 成功条件: 記事ごとの行の合計が、各 cost_total.md に書かれた合計と許容差以内で一致する（check_totals）
- 拒否条件: 記録に無い値（推測・Cost Explorer の値）を ROWS に入れない。合計が合わなければ出力しない
- 1,000枚あたり: 本番の行だけ、費用 ÷ のべ判定数（units）× 1,000。units の数え方は unit_kind と note に書く

使い方: python src/aggregate_costs.py   （記事フォルダで実行。results/cost_by_article.csv と .md を書く）
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

MAIN = "advent-2026（main）"
WT = ".worktrees/{wt}"

NOVA2 = "jp.amazon.nova-2-lite-v1:0"
NOVA2_FLEX = "jp.amazon.nova-2-lite-v1:0（Flex）"
NOVAPRO = "apac.amazon.nova-pro-v1:0"
EMBED = "amazon.nova-2-multimodal-embeddings-v1:0"
HAIKU = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "jp.anthropic.claude-sonnet-4-6"
PLATFORM = "11/30 の構成（モデル＋AgentCore・Step Functions など）"


@dataclass(frozen=True)
class Row:
    article_date: str  # 公開日（MM-DD）
    article: str
    run_date_jst: str
    stage: str  # 試運転 / 本番 / 確認 / なし
    run: str
    model: str
    calls: int | None
    units: int | None  # 1,000枚あたりの分母（のべ判定数）。本番の行だけ
    unit_kind: str
    cost_usd: float
    source: str
    note: str = ""


def per_1000(cost_usd: float, units: int | None) -> float | None:
    """費用 ÷ units × 1,000。units が無い・0以下なら None（試運転などは出さない）。"""
    if cost_usd < 0:
        raise ValueError(f"cost must be >= 0, got {cost_usd}")
    if units is None or units <= 0:
        return None
    return cost_usd / units * 1000


def _src(base: str, path: str) -> str:
    return f"{base}:{path}"


S1111 = _src(MAIN, "november/2026-11-11_aws-zero-training-visa/results/cost_total.md")
S1119 = _src(MAIN, "november/2026-11-19_mmad-remeasure/results/cost_total.md")
S1126 = _src(MAIN, "november/2026-11-26_nova-claude-cascade/results/cost_total.md")
S1130 = _src(WT.format(wt="platform-2026-11-30"), "november/2026-11-30_serverless-inspection-platform/results/cost_total.md")
S1130B = _src(WT.format(wt="platform-2026-11-30"), "november/2026-11-30_serverless-inspection-platform/results/cost_breakdown_per1000.csv")
S1202 = _src(WT.format(wt="localization-2026-12-02"), "advent-calendar/12-02_defect-localization/results/cost_total.md")
S1203 = _src(WT.format(wt="fewshot-2026-12-03"), "advent-calendar/12-03_defect-fewshot/results/cost_total.md")
S1204 = _src(WT.format(wt="cheatsheet-2026-12-04"), "advent-calendar/12-04_reinvent-1_vlm-cheatsheet/results/cost_total.md")
S1205 = _src(WT.format(wt="imagecost-2026-12-05"), "advent-calendar/12-05_image-input-cost/results/cost_total.md")
S1208 = _src(WT.format(wt="flex-2026-12-08"), "advent-calendar/12-08_flex-batch/results/cost_total.md")
S1208R = _src(WT.format(wt="flex-2026-12-08"), "advent-calendar/12-08_flex-batch/results/raw/ac08_flex__night1.jsonl")
S1209 = _src(WT.format(wt="reasoning-2026-12-09"), "advent-calendar/12-09_reasoning-effort/results/cost_total.md")
S1212 = _src(WT.format(wt="calib-2026-12-12"), "advent-calendar/12-12_confidence-calibration/results/cost_total.md")
S1201 = _src(WT.format(wt="yardstick-2026-12-01"), "advent-calendar/12-01_eval-yardstick/results/yardstick_note.md")
S1206 = _src(WT.format(wt="datasets-2026-12-06"), "advent-calendar/12-06_reinvent-2_dataset-guide/results/deviations.md")
S1207 = _src(WT.format(wt="contam-2026-12-07"), "advent-calendar/12-07_mvtec-contamination/results/（空）")
S1210 = _src(WT.format(wt="dino-2026-12-10"), "advent-calendar/12-10_anomalydino-prefilter/results/deviations.md")

A1111, A1119, A1126, A1130 = (
    "aws-zero-training-visa",
    "mmad-remeasure",
    "nova-claude-cascade",
    "serverless-inspection-platform",
)

ROWS: list[Row] = [
    # 11/11（2026-11-05 に実行）
    Row("11-11", A1111, "2026-11-05", "試運転", "n1_A/B/C/wide__trial10", NOVA2, 100, None, "", 0.0377, S1111, "0.0071+0.0100+0.0154+0.0052"),
    Row("11-11", A1111, "2026-11-05", "試運転", "n1_D__trial10", NOVAPRO, 10, None, "", 0.0467, S1111),
    Row("11-11", A1111, "2026-11-05", "試運転", "n1_E__trial10・trial4", EMBED, 14, None, "", 0.0008, S1111, "画像1枚あたりの課金"),
    Row("11-11", A1111, "2026-11-05", "本番", "n1_A", NOVA2, 600, 600, "判定（200枚×3回）", 0.1449, S1111),
    Row("11-11", A1111, "2026-11-05", "本番", "n1_B", NOVA2, 600, 600, "判定（200枚×3回）", 0.1989, S1111),
    Row("11-11", A1111, "2026-11-05", "本番", "n1_C", NOVA2, 600, 600, "判定（200枚×3回）", 0.3085, S1111),
    Row("11-11", A1111, "2026-11-05", "本番", "n1_wide", NOVA2, 120, 120, "判定", 0.0618, S1111),
    Row("11-11", A1111, "2026-11-05", "本番", "n1_D", NOVAPRO, 200, 200, "判定", 1.0393, S1111),
    Row("11-11", A1111, "2026-11-05", "本番", "n1_E", EMBED, 440, 440, "埋め込んだ画像", 0.0264, S1111, "学習用240枚（索引）とテスト200枚。テストだけなら 0.0120 USD"),
    # 11/19（2026-11-05 に実行。単位は問題）
    Row("11-19", A1119, "2026-11-05", "試運転", "m1_haiku45__trial10 ×2", HAIKU, 20, None, "", 0.0354, S1119, "0.0179+0.0175"),
    Row("11-19", A1119, "2026-11-05", "試運転", "m1_nova2lite__trial10 ×2", NOVA2, 20, None, "", 0.0096, S1119, "0.0050+0.0046"),
    Row("11-19", A1119, "2026-11-05", "試運転", "m1_sonnet46__trial10 ×2", SONNET, 20, None, "", 0.0863, S1119, "0.0437+0.0426"),
    Row("11-19", A1119, "2026-11-05", "本番", "m1_haiku45_ondemand", HAIKU, 1050, 1050, "問題（MMAD）", 1.8018, S1119),
    Row("11-19", A1119, "2026-11-05", "本番", "m1_nova2lite", NOVA2, 1050, 1050, "問題（MMAD）", 0.4698, S1119),
    Row("11-19", A1119, "2026-11-05", "本番", "m1_nova2lite_0shot", NOVA2, 1050, 1050, "問題（MMAD）", 0.3079, S1119),
    Row("11-19", A1119, "2026-11-05", "本番", "m1_sonnet46_644", SONNET, 644, 644, "問題（MMAD）", 2.9479, S1119),
    # 11/26（2026-11-11 に実行）
    Row("11-26", A1126, "2026-11-11", "試運転", "n3_claude_refB__trial10", SONNET, 10, None, "", 0.0711, S1126),
    Row("11-26", A1126, "2026-11-11", "本番", "n3_claude_refB", SONNET, 200, 200, "判定", 1.4855, S1126),
    # 11/30（2026-11-20 に実行。2026-11-20 時点の記録。モデル以外も含む）
    Row("11-30", A1130, "2026-11-20", "本番", "P-nova", PLATFORM, 100, 100, "画像", 0.299844, S1130B, "Nova 2 Lite。EventBridge は単価未確認で含まない"),
    Row("11-30", A1130, "2026-11-20", "本番", "P-qwen", PLATFORM, 100, 100, "画像", 0.118381, S1130B, "Qwen3 VL 235B A22B"),
    Row("11-30", A1130, "2026-11-20", "本番", "P-claude", PLATFORM, 50, 50, "画像", 1.960332, S1130B, "Claude Sonnet 4.6（jp.）、キャッシュあり"),
    Row("11-30", A1130, "2026-11-20", "本番", "P-idle-900", PLATFORM, 10, 10, "画像", 0.059882, S1130B, "待ち時間 900秒"),
    Row("11-30", A1130, "2026-11-20", "本番", "P-idle-60", PLATFORM, 10, 10, "画像", 0.032292, S1130B, "待ち時間 60秒"),
    Row("11-30", A1130, "2026-11-20", "本番", "Q（分析エージェント）", PLATFORM, 20, 20, "質問", 0.063673, S1130B),
    Row("11-30", A1130, "2026-11-20", "本番", "共通（ログ・トレース）", PLATFORM, None, None, "", 0.140267, S1130B, "CloudWatch Logs の取り込み 0.137 USD（約95% は送った画像の Base64）"),
    Row("11-30", A1130, "2026-11-20", "試運転", "試運転・スモークテスト・手元（モデル）", PLATFORM, None, None, "", 0.503, S1130),
    Row("11-30", A1130, "2026-11-20", "試運転", "試運転の時間帯の Runtime・Code Interpreter", PLATFORM, None, None, "", 0.012, S1130),
    Row("11-30", A1130, "2026-11-20", "確認", "直したあとの確認（Claude 5枚、モデル）", PLATFORM, 5, None, "", 0.191, S1130),
    # 12/1（Bedrock を呼んでいない）
    Row("12-01", "eval-yardstick", "―", "なし", "記録の読み直しだけ", "―", 0, None, "", 0.0, S1201, "Bedrock は呼んでいない（費用 0 USD）"),
    # 12/2（2026-11-16 に実行）
    Row("12-02", "defect-localization", "2026-11-16", "試運転", "loc_v1・v2・v3（A・B・C）", NOVA2, 270, None, "", 0.150838, S1202),
    Row("12-02", "defect-localization", "2026-11-16", "本番", "A", NOVA2, 420, 420, "画像（1回）", 0.137277, S1202),
    Row("12-02", "defect-localization", "2026-11-16", "本番", "B", NOVA2, 300, 300, "画像（1回）", 0.157254, S1202),
    Row("12-02", "defect-localization", "2026-11-16", "本番", "C", NOVA2, 1680, 420, "画像（4回で1枚）", 0.474426, S1202, "1枚を4回に分けて呼ぶ"),
    # 12/3（2026-11-16 に実行）
    Row("12-03", "defect-fewshot", "2026-11-16", "試運転", "試運転と確認（キャッシュ・画像6枚）", NOVA2, 75, None, "", 0.033110, S1203),
    Row("12-03", "defect-fewshot", "2026-11-16", "本番", "ac03_C0", NOVA2, 400, 400, "判定（200枚×2回）", 0.146192, S1203),
    Row("12-03", "defect-fewshot", "2026-11-16", "本番", "ac03_C1", NOVA2, 400, 400, "判定（200枚×2回）", 0.181340, S1203),
    Row("12-03", "defect-fewshot", "2026-11-16", "本番", "ac03_C3", NOVA2, 400, 400, "判定（200枚×2回）", 0.256081, S1203),
    Row("12-03", "defect-fewshot", "2026-11-16", "本番", "ac03_C3cache", NOVA2, 200, 200, "判定", 0.047057, S1203, "プロンプトキャッシュあり"),
    # 12/4（2026-11-16 に実行。トークン数の確認。呼び出し 63 回のうち失敗 15 回は費用 0）
    *[
        Row("12-04", "reinvent-1_vlm-cheatsheet", "2026-11-16", "確認", "token_check", m, None, None, "", c, S1204, "呼び出しの回数はモデルごとには書かれていない（全体で63回）")
        for m, c in [
            ("global.moonshotai.kimi-k3", 0.010962),
            (SONNET, 0.008557),
            (NOVAPRO, 0.003487),
            (HAIKU, 0.002968),
            ("moonshotai.kimi-k2.5", 0.002208),
            ("mistral.mistral-large-3-675b-instruct", 0.001809),
            ("mistral.magistral-small-2509", 0.001747),
            ("nvidia.nemotron-nano-12b-v2", 0.001631),
            ("qwen.qwen3-vl-235b-a22b", 0.001326),
            ("mistral.ministral-3-14b-instruct", 0.000664),
            ("mistral.ministral-3-8b-instruct", 0.000507),
            (NOVA2, 0.000473),
            ("mistral.ministral-3-3b-instruct", 0.000336),
            ("amazon.nova-lite-v1:0", 0.000250),
            ("google.gemma-3-12b-it", 0.000069),
            ("google.gemma-3-4b-it", 0.000031),
        ]
    ],
    # 12/5（2026-11-16 に実行）
    Row("12-05", "image-input-cost", "2026-11-16", "試運転", "H-512・H-768・H-1568 試運転", HAIKU, 30, None, "", 0.06238, S1205, "0.00908+0.01581+0.03749"),
    Row("12-05", "image-input-cost", "2026-11-16", "試運転", "N-tile 試運転", NOVA2, 10, None, "", 0.00623, S1205),
    Row("12-05", "image-input-cost", "2026-11-16", "本番", "N-tile", NOVA2, 200, 200, "判定", 0.12464, S1205),
    Row("12-05", "image-input-cost", "2026-11-16", "本番", "H-512", HAIKU, 200, 200, "判定", 0.18064, S1205),
    Row("12-05", "image-input-cost", "2026-11-16", "本番", "H-768", HAIKU, 200, 200, "判定", 0.31176, S1205),
    Row("12-05", "image-input-cost", "2026-11-16", "本番", "H-1568", HAIKU, 200, 200, "判定", 0.74091, S1205),
    # 12/6（AWS を呼んでいない）
    Row("12-06", "reinvent-2_dataset-guide", "―", "なし", "データセットの調べもの", "―", 0, None, "", 0.0, S1206, "AWS の API・CLI は呼んでいない"),
    # 12/7（この集計の時点で記録なし）
    Row("12-07", "mvtec-contamination", "―", "なし", "（記録なし）", "―", None, None, "", 0.0, S1207, "2026-11-21 の時点で results/ に費用の記録が無い。検証が済んだら足す"),
    # 12/8（Standard は 2026-11-16、Flex の夜の続きは 2026-11-21）
    Row("12-08", "flex-batch", "2026-11-16", "試運転", "ac08_standard__trial10", NOVA2, 10, None, "", 0.003302, S1208),
    Row("12-08", "flex-batch", "2026-11-16", "試運転", "ac08_flex__trial10", NOVA2_FLEX, 10, None, "", 0.001651, S1208),
    Row("12-08", "flex-batch", "2026-11-16", "本番", "ac08_standard__day1", NOVA2, 200, 200, "判定", 0.066285, S1208),
    Row("12-08", "flex-batch", "2026-11-16", "本番", "ac08_flex__day1", NOVA2_FLEX, 200, 200, "判定", 0.033114, S1208),
    Row("12-08", "flex-batch", "2026-11-16", "本番", "ac08_standard__night1", NOVA2, 200, 200, "判定", 0.066169, S1208),
    Row("12-08", "flex-batch", "2026-11-16", "本番", "ac08_flex__night1（11/16 の73件）", NOVA2_FLEX, 73, 73, "判定", 0.012109, S1208R, "cost_usd を timestamp の日付で分けた"),
    Row("12-08", "flex-batch", "2026-11-21", "本番", "ac08_flex__night1（11/21 の続き127件）", NOVA2_FLEX, 127, 127, "判定", 0.021000, S1208R, "同上。Cost Explorer の照合はまだ"),
    # 12/9（2026-11-16 に実行）
    Row("12-09", "reasoning-effort", "2026-11-16", "試運転", "ac09_*__trial5", NOVA2, 25, None, "", 0.0891, S1209, "0.0017+0.0140+0.0087+0.0647（cost_total.md の小計は 0.0889。4桁の丸めの差）"),
    Row("12-09", "reasoning-effort", "2026-11-16", "本番", "ac09_low", NOVA2, 400, 400, "判定（200枚×2回）", 0.6147, S1209, "推論 low"),
    Row("12-09", "reasoning-effort", "2026-11-16", "本番", "ac09_med", NOVA2, 200, 200, "判定", 0.4021, S1209, "推論 medium"),
    Row("12-09", "reasoning-effort", "2026-11-16", "本番", "ac09_high20", NOVA2, 20, 20, "判定", 0.3184, S1209, "推論 high"),
    # 12/10（Bedrock を呼んでいない）
    Row("12-10", "anomalydino-prefilter", "―", "なし", "AnomalyDINO（手元の CPU）", "―", 0, None, "", 0.0, S1210, "Bedrock は呼んでいない（AWS の費用 0）。2段目は 11/11・11/26 の記録を使い回し"),
    # 12/12（2026-11-16 に実行）
    Row("12-12", "confidence-calibration", "2026-11-16", "試運転", "ac12_t07__trial10", NOVA2, 50, None, "", 0.0166, S1212),
    Row("12-12", "confidence-calibration", "2026-11-16", "本番", "ac12_t07（200枚×5回、エラー2回の呼び直しを含む）", NOVA2, 1002, 1000, "判定（200枚×5回）", 0.3419, S1212, "0.3412+0.0007"),
]

# 各 cost_total.md に書かれた記事の合計（照合用）。12/1・12/6・12/10 は記録で 0、12/7 は記録なし
STATED_TOTALS: dict[str, float] = {
    "11-11": 1.8650,
    "11-19": 5.6588,
    "11-26": 1.5566,
    "11-30": 3.38,  # 「約 3.38」
    "12-01": 0.0,
    "12-02": 0.919796,
    "12-03": 0.66378,
    "12-04": 0.0370,
    "12-05": 1.4266,
    "12-06": 0.0,
    "12-07": 0.0,
    "12-08": 0.203629,
    "12-09": 1.4242,
    "12-10": 0.0,
    "12-12": 0.3585,
}
TOLERANCE = {"11-30": 0.005}  # 「約 3.38」は小数2桁
DEFAULT_TOLERANCE = 0.0003  # 4桁で丸めた行を足したときの差


def article_totals(rows: list[Row]) -> "OrderedDict[str, float]":
    out: OrderedDict[str, float] = OrderedDict()
    for r in rows:
        out[r.article_date] = out.get(r.article_date, 0.0) + r.cost_usd
    return out


def check_totals(rows: list[Row], stated: dict[str, float]) -> list[str]:
    """記事ごとの合計が cost_total.md の合計と合わない記事を返す（空なら合格）。"""
    got = article_totals(rows)
    bad = []
    for date in sorted(set(got) | set(stated)):
        if date not in got or date not in stated:
            bad.append(f"{date}: rows={got.get(date)} stated={stated.get(date)}")
            continue
        tol = TOLERANCE.get(date, DEFAULT_TOLERANCE)
        if abs(got[date] - stated[date]) > tol:
            bad.append(f"{date}: rows={got[date]:.6f} stated={stated[date]:.6f}")
    return bad


def model_totals(rows: list[Row]) -> "OrderedDict[str, tuple[float, int]]":
    """モデルごとの費用と、呼び出し回数（分かる行だけ）の合計。費用の大きい順。"""
    acc: dict[str, list] = {}
    for r in rows:
        if r.cost_usd == 0:
            continue
        a = acc.setdefault(r.model, [0.0, 0])
        a[0] += r.cost_usd
        a[1] += r.calls or 0
    return OrderedDict(sorted(((k, (v[0], v[1])) for k, v in acc.items()), key=lambda kv: -kv[1][0]))


def window_total(rows: list[Row], dates: set[str]) -> float:
    return sum(r.cost_usd for r in rows if r.article_date in dates)


def run_date_total(rows: list[Row], run_dates: set[str]) -> float:
    return sum(r.cost_usd for r in rows if r.run_date_jst in run_dates)


def _fmt(x: float | None, nd: int = 4) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def write_csv(rows: list[Row], path: Path) -> None:
    cols = ["article_date", "article", "run_date_jst", "stage", "run", "model", "calls", "units", "unit_kind",
            "cost_usd", "per_1000_usd", "source", "note"]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            p = per_1000(r.cost_usd, r.units) if r.stage == "本番" else None
            w.writerow([r.article_date, r.article, r.run_date_jst, r.stage, r.run, r.model,
                        "" if r.calls is None else r.calls, "" if r.units is None else r.units, r.unit_kind,
                        f"{r.cost_usd:.6f}", _fmt(p, 4), r.source, r.note])


def main() -> None:
    bad = check_totals(ROWS, STATED_TOTALS)
    if bad:
        raise SystemExit("合計が cost_total.md と合わない: " + "; ".join(bad))
    RESULTS.mkdir(exist_ok=True)
    write_csv(ROWS, RESULTS / "cost_by_article.csv")
    tot = article_totals(ROWS)
    for d, v in tot.items():
        prod = [per_1000(r.cost_usd, r.units) for r in ROWS if r.article_date == d and r.stage == "本番" and r.units]
        rng = f"{min(prod):.3f}〜{max(prod):.3f}" if prod else "―"
        print(f"{d}\t{v:.4f}\t本番の1,000あたり {rng}")
    nov = window_total(ROWS, {"11-11", "11-19", "11-26", "11-30"})
    dec10 = window_total(ROWS, {f"12-{i:02d}" for i in range(1, 11)})
    print(f"11月の4本 {nov:.4f} / 12/1〜12/10 {dec10:.4f} / 12/12 {window_total(ROWS, {'12-12'}):.4f} / 全部 {sum(tot.values()):.4f}")
    # Cost Explorer の 11/20 時点の値と比べる範囲（実行日が 11/20 より前の記録）
    pre = run_date_total(ROWS, {"2026-11-05", "2026-11-11", "2026-11-16"})
    print(f"実行日が 11/5・11/11・11/16 の記録の合計 {pre:.4f}")
    for m, (c, n) in model_totals(ROWS).items():
        print(f"{m}\t{c:.4f}\t{n}")


if __name__ == "__main__":
    main()
