"""検査エージェントの前処理ツール。処理は AgentCore Code Interpreter（組み込みの aws.codeinterpreter.v1）の中で OpenCV で行う。

- モデルに任意のコードは書かせない。決まった処理（zoom・contrast・diff）と数値の引数だけを選ばせ、
  コードはここで組み立てる（引数は検証してから JSON の値として埋め込む）
- Code Interpreter のセッションは画像1枚ごとに開き、判定のあとで閉じる（close() を必ず呼ぶ）
- 呼び出しの回数に上限を設け、超えたらモデルに判定を促す（費用の上限）
- モデルが1回の応答でツールを並列に呼んでも（Claude は2つ同時がほとんど）、1枚の中では順に実行し、
  Code Interpreter のセッションは1つだけ開く。2026-11-20 の本番（P-claude）では、並列に呼ばれたときに
  セッションを2つ開いて1つしか閉じず、残りが上限の300秒まで課金されていた（deviations.md の 12）
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Protocol

OPS = ("zoom", "contrast", "diff")
OUT_LONG_EDGE = 1024

_CODE = r"""
import json, cv2
p = json.loads(PARAMS)
t = cv2.imread("target.img", cv2.IMREAD_COLOR)
if t is None:
    raise SystemExit("target image could not be read")
h, w = t.shape[:2]
if p["op"] == "zoom":
    x0, y0 = int(p["x"] * w), int(p["y"] * h)
    x1, y1 = max(x0 + 1, int((p["x"] + p["w"]) * w)), max(y0 + 1, int((p["y"] + p["h"]) * h))
    out = t[y0:y1, x0:x1]
elif p["op"] == "contrast":
    lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
else:
    r = cv2.imread("ref.img", cv2.IMREAD_COLOR)
    r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
    d = cv2.absdiff(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    out = cv2.applyColorMap(d, cv2.COLORMAP_JET)
oh, ow = out.shape[:2]
s = OUT_LONG_EDGE / max(oh, ow)
out = cv2.resize(out, (max(1, round(ow * s)), max(1, round(oh * s))),
                 interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
cv2.imwrite("out.png", out)
print(json.dumps({"op": p["op"], "width": out.shape[1], "height": out.shape[0]}))
"""


def validate_params(op: str, x: float = 0.0, y: float = 0.0, w: float = 1.0, h: float = 1.0) -> dict[str, Any]:
    """モデルが選んだ引数を検証する。範囲外は ValueError（コードを作らない）。"""
    if op not in OPS:
        raise ValueError(f"op must be one of {OPS}: {op!r}")
    vals = {"x": x, "y": y, "w": w, "h": h}
    for k, v in vals.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{k} must be a number: {v!r}")
    if op == "zoom":
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.0001 and y + h <= 1.0001):
            raise ValueError("zoom region must be inside the image (fractions 0..1)")
        if w < 0.05 or h < 0.05:
            raise ValueError("zoom region is too small (w and h must be >= 0.05)")
        return {"op": op, "x": float(x), "y": float(y), "w": float(w), "h": float(h)}
    return {"op": op}


def build_code(params: dict[str, Any]) -> str:
    """検証済みの引数を JSON の文字列リテラルとして埋め込む。"""
    return (f"PARAMS = {json.dumps(json.dumps(params))}\nOUT_LONG_EDGE = {OUT_LONG_EDGE}\n" + _CODE)


class Sandbox(Protocol):
    def start(self, identifier: str | None = ..., name: str | None = ..., session_timeout_seconds: int | None = ...) -> str: ...
    def upload_file(self, path: str, content: bytes | str, description: str = "") -> dict: ...
    def execute_code(self, code: str, language: str = "python", clear_context: bool = False) -> dict: ...
    def download_file(self, path: str) -> bytes | str: ...
    def stop(self) -> bool: ...


def _stdout_and_error(result: dict) -> tuple[str, str | None]:
    """executeCode の応答（stream）から標準出力とエラーを取り出す。"""
    out, err = [], None
    for event in result.get("stream", []):
        res = event.get("result") or {}
        sc = res.get("structuredContent") or {}
        if sc.get("stdout"):
            out.append(sc["stdout"])
        if res.get("isError") or sc.get("exitCode") not in (None, 0):
            err = sc.get("stderr") or json.dumps(res.get("content", []), ensure_ascii=False)[:500]
    return "".join(out), err


class Preprocessor:
    """画像1枚ぶんの前処理ツール。Strands のツールとして tool() を渡す。"""

    def __init__(self, sandbox_factory: Callable[[], Sandbox], target: bytes, ref: bytes | None,
                 max_calls: int, session_timeout_seconds: int = 300):
        self._factory = sandbox_factory
        self._target, self._ref = target, ref
        self.max_calls = max_calls
        self.calls = 0
        self.sandbox_used = False
        self._sb: Sandbox | None = None
        self._timeout = session_timeout_seconds
        self._lock = threading.Lock()  # 並列のツール呼び出しを1枚の中で順にする

    def _sandbox(self) -> Sandbox:
        if self._sb is None:
            sb = self._factory()
            sb.start(session_timeout_seconds=self._timeout)
            sb.upload_file("target.img", self._target)
            if self._ref is not None:
                sb.upload_file("ref.img", self._ref)
            self._sb = sb
            self.sandbox_used = True
        return self._sb

    def run(self, op: str, x: float = 0.0, y: float = 0.0, w: float = 1.0, h: float = 1.0) -> dict[str, Any]:
        """Strands のツールの本体。ToolResult の形（status と content）で返す。"""
        with self._lock:
            return self._run(op, x, y, w, h)

    def _run(self, op: str, x: float, y: float, w: float, h: float) -> dict[str, Any]:
        if self.calls >= self.max_calls:
            return {"status": "error", "content": [{"text": f"preprocess は {self.max_calls} 回までです。これまでの画像で判定してください。"}]}
        self.calls += 1
        try:
            params = validate_params(op, x, y, w, h)
            if params["op"] == "diff" and self._ref is None:
                raise ValueError("diff needs a reference image")
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"引数が正しくありません: {e}"}]}
        sb = self._sandbox()
        out, err = _stdout_and_error(sb.execute_code(build_code(params)))
        if err:
            return {"status": "error", "content": [{"text": f"前処理に失敗しました: {err}"}]}
        png = sb.download_file("out.png")
        if isinstance(png, str):
            return {"status": "error", "content": [{"text": "前処理の結果を画像として読めませんでした"}]}
        return {"status": "success", "content": [
            {"text": f"{params['op']} の結果です。{out.strip()}"},
            {"image": {"format": "png", "source": {"bytes": png}}},
        ]}

    def close(self) -> None:
        with self._lock:
            if self._sb is not None:
                try:
                    self._sb.stop()
                finally:
                    self._sb = None
