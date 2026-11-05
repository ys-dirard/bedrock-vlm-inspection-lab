"""プロンプトの組み立て（CI-MP1〜MP3）。gpt4o.py の get_query・parse_conversation と同じ文字列を作る。

gpt4o.py の組み立て（commit 49db95b）:
    instruction + incontext + "The last image is the query image" + "Following is the question list: " + "\\n" + conversation_text
- incontext は見本があるとき "The first {n} image is the normal sample, which can be used as a template to compare."
  （"compare." と "The last" の間に空白が無いのも、そのまま再現する）
- conversation_text は、1〜i 問目について f"Question: {問題文} \\n{選択肢}" + "\\n"。選択肢は f"{記号}. {文}\\n" を mmad.json の順に並べる
- 画像は [見本…, 検査対象] の順で、文の前に置く
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from ..data import EncodedImage

INSTRUCTION_FILE = "prompts/mmad_gpt4o_instruction.txt"


def load_instruction(base_dir: str | Path) -> str:
    return (Path(base_dir) / INSTRUCTION_FILE).read_text(encoding="utf-8")


def question_text(q: dict[str, Any]) -> str:
    options = "".join(f"{k}. {v}\n" for k, v in q["Options"].items())
    return f"Question: {q['Question']} \n{options}"


def build_text(instruction: str, conversation: list[dict[str, Any]], q_index: int, n_shots: int) -> str:
    """q_index 問目を聞くときの文。1〜q_index 問目の問題文を並べる（前の答えは入れない）。"""
    if not 0 <= q_index < len(conversation):
        raise IndexError(f"q_index {q_index} out of range for {len(conversation)} questions")
    if n_shots < 0:
        raise ValueError("n_shots must be >= 0")
    incontext = (
        f"The first {n_shots} image is the normal sample, which can be used as a template to compare."
        if n_shots > 0 else ""
    )
    conversation_text = "".join(question_text(q) + "\n" for q in conversation[: q_index + 1])
    return (
        instruction + incontext + "The last image is the query image"
        + "Following is the question list: " + "\n" + conversation_text
    )


def anthropic_body(images: list[EncodedImage], text: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    """Claude の Messages API の本文（InvokeModel とバッチ推論で同じものを使う）。画像は [見本…, 検査対象] の順。"""
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{img.format}",
                "data": base64.b64encode(img.data).decode("ascii"),
            },
        }
        for img in images
    ]
    content.append({"type": "text", "text": text})
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
