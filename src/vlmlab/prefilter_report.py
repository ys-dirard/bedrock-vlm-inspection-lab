"""2段構成（prefilter.py）の結果を書き出す：表1（CSV と Markdown）、画像ごとの異常度、図（fig_prefilter.png）。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

MAIN_SCOPE = "category"
NOTE_LABELS = ("「正解でしきい値」が「はい」の行は、テスト画像の正解を見てしきい値を決めた楽観的な値です（同じ200枚でしきい値を決めて、"
               "同じ200枚で評価している）。固定の割合（上位 X%、残りは OK）の行は正解を使っていません。")
NOTE_COST = ("費用は 11/11 の条件 B の記録の cost_usd（jp.amazon.nova-2-lite-v1:0、東京、Standard。入力 0.396・出力 3.311 USD / 100万トークン）。"
             "主の表は repeat=1 の1回分、多数決の表は3回分。AnomalyDINO（手元の CPU）の費用は 0 として数えています（電気代・PC の費用は数えない）。")
SCOPE_JA = {"category": "品目ごと", "global": "全体"}
REST_JA = {"OK": "OK", "AD_miss2": "AnomalyDINO（見逃し2%）", "AD_miss5": "AnomalyDINO（見逃し5%）"}


def _fmt(v: Any, pct: bool = False, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.{digits}f}"


def _pick(rows: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def main_rows(rows: list[dict[str, Any]], scope: str = MAIN_SCOPE) -> list[dict[str, Any]]:
    """表1の主の行：Nova だけ、AnomalyDINO だけ、しきい値で回す、上位 X% を回す（残りは OK）。"""
    return (_pick(rows, routing="all") + _pick(rows, routing="none", scope=scope)
            + _pick(rows, routing="threshold", scope=scope) + _pick(rows, routing="top", scope=scope, rest="OK"))


def _md_table(rows: list[dict[str, Any]], maj: bool = False) -> list[str]:
    head = ["条件", "範囲", "回さない画像", "回した割合", "見逃し率", "誤検知率", "F1", "1,000枚あたり（USD）", "正解でしきい値"]
    p = "maj_" if maj else ""
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in rows:
        rest = "-" if r["routing"] == "all" else REST_JA.get(r["rest"], r["rest"])
        lines.append("| " + " | ".join([
            r["condition"], SCOPE_JA.get(r["scope"], r["scope"]), rest,
            _fmt(r["send_rate"], pct=True), _fmt(r[f"{p}miss_rate"], pct=True),
            _fmt(r[f"{p}false_positive_rate"], pct=True), _fmt(r[f"{p}f1"]),
            _fmt(r[f"{p}cost_per_1000_usd"], digits=4), "はい" if r["threshold_from_labels"] else "いいえ",
        ]) + " |")
    return lines


def write_outputs(results: dict[str, dict[str, Any]], ad_runs: dict[str, pd.DataFrame],
                  times: dict[str, dict[str, Any]], out_dir: str | Path,
                  alt_rows: list[dict[str, Any]] | None = None) -> dict[str, Path]:
    """results・ad_runs・times のキーは解像度の名前（先頭が主）。alt_rows は2段目を替えた参考の行（CI-F10、主の解像度）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(results)
    main = names[0]
    paths: dict[str, Path] = {}

    table = results[main]["table"]
    base = table[["category", "label"]].copy()
    for n in names:
        a = ad_runs[n].set_index("image_id")
        base[f"score_{n}"] = a["score"]
        base[f"time_s_{n}"] = a["time_s"]
        base[f"n_foreground_{n}"] = a["n_foreground"]
    base["nova_r1_pred"] = table["nova1_pred"]
    base["nova_maj_pred"] = table["novamaj_pred"]
    paths["scores"] = out / "anomalydino_scores.csv"
    base.rename_axis("image_id").reset_index().to_csv(paths["scores"], index=False, lineterminator="\n")

    paths["summary_csv"] = out / "prefilter_summary.csv"
    csv_rows = [{"table": "main", "resolution": n, "stage2": NOVA_LABEL, **r} for n in names for r in results[n]["rows"]]
    csv_rows += [{"table": "alt_stage2", "resolution": main, **r} for r in (alt_rows or [])]
    pd.DataFrame(csv_rows).to_csv(paths["summary_csv"], index=False, lineterminator="\n")
    paths["score_summary"] = out / "anomalydino_score_summary.csv"
    pd.concat([results[n]["scores"].assign(resolution=n) for n in names]).to_csv(
        paths["score_summary"], index=False, lineterminator="\n")
    paths["time"] = out / "anomalydino_time.json"
    paths["time"].write_text(json.dumps(times, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = results[main]["rows"]
    md = [f"# 表1：2段構成（AnomalyDINO → Nova 2 Lite）", "",
          "- テスト：VisA の pcb1・capsules、良品100枚・不良品100枚（visa_test_200.csv）。AnomalyDINO の見本は品目ごとに良品1枚（11/11 と同じ）",
          "- 回した画像は Nova 2 Lite の判定（主は 11/11 の条件 B の repeat=1）。Nova が解析失敗の画像と回さなかった画像は「回さない画像」の列の扱い",
          "- " + NOTE_LABELS, "- " + NOTE_COST,
          "", f"## 主：解像度 {main}・品目ごとに上位 X%・Nova は1回", ""]
    md += _md_table(main_rows(rows))
    md += ["", f"## 参考：回さなかった画像を AnomalyDINO の判定にする（解像度 {main}・品目ごと）", ""]
    md += _md_table(_pick(rows, routing="top", scope=MAIN_SCOPE, rest="AD_miss2")
                    + _pick(rows, routing="top", scope=MAIN_SCOPE, rest="AD_miss5"))
    md += ["", f"## 参考：200枚全体で上位 X%・しきい値を取る（解像度 {main}）", ""]
    md += _md_table(main_rows(rows, scope="global")[1:])
    md += ["", f"## 参考：Nova の3回の多数決（費用は3回分。解像度 {main}・品目ごと）", ""]
    md += _md_table(main_rows(rows), maj=True)
    for n in names[1:]:
        md += ["", f"## 参考：解像度 {n}（品目ごと・Nova は1回）", ""]
        md += _md_table(main_rows(results[n]["rows"])[1:])
    if alt_rows:
        md += ["", f"## 参考：2段目を替えた場合（解像度 {main}・品目ごとに上位 X%・2段目は1回）", "",
               "- 2段目は記録から組み合わせた（API は呼んでいない）。" + "、".join(ALT_NOTES.get(s, s) for s in _stages(alt_rows)),
               "- 回した画像で2段目が解析失敗なら「回さない画像」と同じ扱い（11/26 の段階判定と同じ考え方）。"
               "「2段目だけ」の行は解析失敗を指標から除き、有効件数を書いた（11/26 の「Claude だけ」と同じ）。費用は解析失敗も含めて回した分",
               "- 「正解でしきい値」が「はい」の行は、テスト画像の正解で決めたしきい値を使った楽観的な値", ""]
        md += _md_alt_table(alt_rows)
    md += ["", "## AnomalyDINO の異常度（品目ごと）", "",
           "| 解像度 | 品目 | AUROC | 良品の中央値 | 良品の範囲 | 不良品の中央値 | 不良品の範囲 |", "|" + " --- |" * 7]
    for n in names:
        for _, s in results[n]["scores"].iterrows():
            md.append(f"| {n} | {s['category']} | {_fmt(s['auroc'])} | {_fmt(s['normal_median'])} | "
                      f"{_fmt(s['normal_min'])}〜{_fmt(s['normal_max'])} | {_fmt(s['anomaly_median'])} | "
                      f"{_fmt(s['anomaly_min'])}〜{_fmt(s['anomaly_max'])} |")
    md += ["", "## 手元の時間（CPU。ほかの検証と同時に動かした値）", "",
           "| 解像度 | 枚数 | 1枚の p50（秒） | p95（秒） | 平均（秒） | 合計（秒） | 見本の特徴づくり（秒） | 最大メモリ（MB） |",
           "|" + " --- |" * 8]
    for n in names:
        t = times[n]
        bank = "、".join(f"{c} {v:.1f}" for c, v in (t.get("bank_time_s") or {}).items())
        md.append(f"| {n} | {t['n']} | {t['time_p50_s']:.2f} | {t['time_p95_s']:.2f} | {t['time_mean_s']:.2f} | "
                  f"{t['time_total_s']:.0f} | {bank} | {_fmt(t.get('peak_memory_mb'), digits=0)} |")
    paths["summary_md"] = out / "prefilter_summary.md"
    paths["summary_md"].write_text("\n".join(md) + "\n", encoding="utf-8")

    paths["fig"] = out / "fig_prefilter.png"
    plot(rows, paths["fig"], main)
    if alt_rows:
        paths["fig_alt"] = out / "fig_prefilter_alt.png"
        plot_alt(alt_rows, _pick(rows, routing="none", scope=MAIN_SCOPE, rest="AD_miss5")[0], paths["fig_alt"], main)
    return paths


NOVA_LABEL = "Nova 2 Lite"
ALT_NOTES = {
    "Nova 2 Lite": "Nova 2 Lite = 11/11 の条件 B の repeat=1（主の表と同じ）",
    "Nova Pro": "Nova Pro = 11/11 の条件 D（apac.amazon.nova-pro-v1:0、AWS サンプルの指示、長辺1024px、1回）",
    "Claude Sonnet 4.6": "Claude Sonnet 4.6 = 11/26 の n3_claude_refB（jp.anthropic.claude-sonnet-4-6、条件 B と同じ指示と見本、長辺768px、1回）",
}
STAGE_COLORS = {"Nova 2 Lite": "#2a78d6", "Nova Pro": "#eb6834", "Claude Sonnet 4.6": "#1baf7a"}
STAGE_MARKERS = {"Nova 2 Lite": "o", "Nova Pro": "s", "Claude Sonnet 4.6": "D"}


def _stages(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(r["stage2"] for r in rows))


def _md_alt_table(rows: list[dict[str, Any]]) -> list[str]:
    head = ["2段目", "条件", "回さない画像", "回した割合", "有効件数", "見逃し率", "誤検知率", "F1", "1,000枚あたり（USD）",
            "正解でしきい値"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in rows:
        rest = "-（解析失敗は除く）" if r["routing"] == "all" else REST_JA.get(r["rest"], r["rest"])
        lines.append("| " + " | ".join([
            r["stage2"], r["condition"], rest, _fmt(r["send_rate"], pct=True), str(r["n_valid"]),
            _fmt(r["miss_rate"], pct=True), _fmt(r["false_positive_rate"], pct=True), _fmt(r["f1"]),
            _fmt(r["cost_per_1000_usd"], digits=4), "はい" if r["threshold_from_labels"] else "いいえ",
        ]) + " |")
    return lines


def _setup_axes(plt, axes, ink, muted, grid):
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)
        ax.set_xlim(-5, 108)
        ax.set_xlabel("2段目に回した割合（%）", color=ink, fontsize=10)


def _japanese_font():
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Yu Gothic", "Meiryo", "BIZ UDGothic", "Noto Sans JP") if f in names), None)
    if font is None:
        raise RuntimeError("no Japanese font found")
    return font


def plot_alt(rows: list[dict[str, Any]], ad_only: dict[str, Any], path: Path, resolution: str) -> None:
    """2段目ごとに、回した割合（横軸）と見逃し率（回さない画像は OK／AnomalyDINO の判定）・費用。3枚、軸は1枚に1つ。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": _japanese_font(), "axes.unicode_minus": False})
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3de"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), dpi=150, facecolor="white")
    _setup_axes(plt, axes, ink, muted, grid)
    panels = [(axes[0], "OK", "miss_rate", 100, "見逃し率（回さない画像は OK）", "見逃し率（%）"),
              (axes[1], "AD_miss5", "miss_rate", 100, "見逃し率（回さない画像は AnomalyDINO の判定※）", "見逃し率（%）"),
              (axes[2], "OK", "cost_per_1000_usd", 1, "Bedrock の費用（2段目は1回分）", "1,000枚あたりの費用（USD）")]
    for ax, rest, key, scale, title, ylabel in panels:
        for s in _stages(rows):
            c, mk = STAGE_COLORS.get(s, ink), STAGE_MARKERS.get(s, "o")
            top = sorted(_pick(rows, stage2=s, routing="top", rest=rest), key=lambda r: r["x_pct"])
            only = _pick(rows, stage2=s, routing="all")[0]
            xs = [0] if key == "miss_rate" and rest != "OK" else []
            ys = [ad_only["miss_rate"] * scale] if xs else []
            xs += [r["send_rate"] * 100 for r in top] + [100]
            ys += [r[key] * scale for r in top] + [only[key] * scale]
            ax.plot(xs, ys, color=c, lw=2, marker=mk, ms=7, label=f"2段目：{s}")
            ax.annotate(f"{only[key] * scale:.3f} USD" if key == "cost_per_1000_usd" else f"{only[key] * scale:.0f}%",
                        (100, only[key] * scale), textcoords="offset points", xytext=(-8, 6), ha="right",
                        fontsize=8, color=ink)
        if key == "miss_rate":
            ax.plot([0], [ad_only["miss_rate"] * 100], color="#eda100", marker="^", ms=10, ls="none",
                    markeredgecolor=ink, markeredgewidth=0.6, label="AnomalyDINO だけ（※）")
            ax.set_ylim(-3, 105)
        else:
            ax.plot([0], [0], color="#eda100", marker="^", ms=10, ls="none", markeredgecolor=ink, markeredgewidth=0.6)
            ax.set_ylim(bottom=-0.2)
        ax.set_ylabel(ylabel, color=ink, fontsize=10)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9, labelcolor=ink,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.01, 0.01, f"※ 見逃し率5%のしきい値（テスト画像の正解で決めた楽観的な値）。右端（100%）は2段目だけ（解析失敗は除く）。"
             f"AnomalyDINO は解像度{resolution}、品目ごとに上位 10・20・30・50%。データ: VisA（CC BY 4.0）の pcb1・capsules 200枚",
             fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot(rows: list[dict[str, Any]], path: Path, resolution: str) -> None:
    """回した割合（横軸）と、見逃し率・1,000枚あたりの費用。軸は1枚に1つ（2枚に分ける）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Yu Gothic", "Meiryo", "BIZ UDGothic", "Noto Sans JP") if f in names), None)
    if font is None:
        raise RuntimeError("no Japanese font found")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3de"
    c_ok, c_ad, c_nova, c_adonly = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

    top_ok = sorted(_pick(rows, routing="top", scope=MAIN_SCOPE, rest="OK"), key=lambda r: r["x_pct"])
    top_ad = sorted(_pick(rows, routing="top", scope=MAIN_SCOPE, rest="AD_miss5"), key=lambda r: r["x_pct"])
    nova = _pick(rows, routing="all")[0]
    ad5 = _pick(rows, routing="none", scope=MAIN_SCOPE, rest="AD_miss5")[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)
        ax.set_xlim(-5, 108)
        ax.set_xlabel("Nova 2 Lite に回した割合（%）", color=ink, fontsize=10)

    ax = axes[0]
    x = [r["send_rate"] * 100 for r in top_ok]
    ax.plot(x, [r["miss_rate"] * 100 for r in top_ok], color=c_ok, lw=2, marker="o", ms=7,
            label="2段：上位 X% を回す（回さない画像は OK）")
    ax.plot(x, [r["miss_rate"] * 100 for r in top_ad], color=c_ad, lw=2, marker="s", ms=7,
            label="2段：上位 X% を回す（回さない画像は AnomalyDINO の判定※）")
    ax.plot([100], [nova["miss_rate"] * 100], color=c_nova, marker="D", ms=9, ls="none",
            label="Nova 2 Lite だけ（全件）")
    ax.plot([0], [ad5["miss_rate"] * 100], color=c_adonly, marker="^", ms=10, ls="none",
            markeredgecolor=ink, markeredgewidth=0.6, label="AnomalyDINO だけ（※）")
    ax.set_ylim(-3, 105)
    ax.set_ylabel("見逃し率（%）", color=ink, fontsize=10)
    ax.set_title("見逃し率（不良品のうち OK とした割合）", color=ink, fontsize=11, loc="left")

    ax = axes[1]
    ax.plot(x, [r["cost_per_1000_usd"] for r in top_ok], color=c_ok, lw=2, marker="o", ms=7)
    ax.plot([100], [nova["cost_per_1000_usd"]], color=c_nova, marker="D", ms=9, ls="none")
    ax.plot([0], [0], color=c_adonly, marker="^", ms=10, ls="none", markeredgecolor=ink, markeredgewidth=0.6)
    ax.annotate(f"{nova['cost_per_1000_usd']:.3f} USD", (100, nova["cost_per_1000_usd"]), textcoords="offset points",
                xytext=(-8, 8), ha="right", fontsize=8, color=ink)
    ax.annotate("0 USD（手元）", (0, 0), textcoords="offset points", xytext=(8, 6), fontsize=8, color=ink)
    for r in top_ok:
        ax.annotate(f"上位{r['x_pct']}%", (r["send_rate"] * 100, r["cost_per_1000_usd"]), textcoords="offset points",
                    xytext=(-6, 8), ha="right", fontsize=8, color=muted)
    ax.set_ylim(bottom=-0.01)
    ax.set_ylabel("1,000枚あたりの費用（USD）", color=ink, fontsize=10)
    ax.set_title("Bedrock の費用（Nova は1回分）", color=ink, fontsize=11, loc="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=ink,
               bbox_to_anchor=(0.5, 0.045))
    fig.text(0.01, 0.01, f"※ 見逃し率5%のしきい値（テスト画像の正解で決めた楽観的な値）。AnomalyDINO は解像度{resolution}、"
             "品目ごとに上位 X%。データ: VisA（CC BY 4.0）の pcb1・capsules 200枚", fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)
