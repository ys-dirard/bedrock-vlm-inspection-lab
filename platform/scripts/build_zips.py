"""エージェントの ZIP（AgentCore Runtime の直接のデプロイ用）と、Lambda のコードを作る。

- 依存は Linux arm64・Python 3.12 の wheel だけを集める（uv pip install --python-platform aarch64-manylinux2014）
  https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-code-deploy-python.html （2026-11-20 確認）
- 手元は Windows なので、ZIP の中の権限（ファイル 644、ディレクトリ 755）を zipfile で明示する
- 上限は 250 MB（ZIP）／750 MB（展開後）。超えたら止める

  uv run python scripts/build_zips.py          # platform/ で実行。dist/ に書く
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
BUILD, DIST = ROOT / "build", ROOT / "dist"
ZIP_LIMIT, UNZIPPED_LIMIT = 250 * 1024 * 1024, 750 * 1024 * 1024

# 11月のコードは書き換えずに、そのまま ZIP に入れる（parse と cost だけ）
VLMLAB_FILES = ["__init__.py", "parse.py", "cost.py"]

AGENTS = {
    "inspector": ["main.py", "judge.py", "preprocess.py"],
    "analyst": ["main.py"],
}


def copy_shared(dest: Path, with_vlmlab: bool = True) -> None:
    shutil.copytree(ROOT / "src" / "platformlab", dest / "platformlab",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if with_vlmlab:
        (dest / "vlmlab").mkdir()
        for f in VLMLAB_FILES:
            shutil.copy2(LAB / "src" / "vlmlab" / f, dest / "vlmlab" / f)


def install_deps(req: Path, dest: Path) -> None:
    cmd = ["uv", "pip", "install", "--python-platform", "aarch64-manylinux2014", "--python-version", "3.12",
           "--target", str(dest), "--only-binary=:all:", "-r", str(req)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


OTEL_SCRIPT = """#!/usr/bin/env python3
# Linux 用のコンソールスクリプト（Windows の uv が作る .exe の代わり。scripts/build_zips.py が書く）
import sys
from opentelemetry.instrumentation.auto_instrumentation import run
sys.exit(run())
"""


def fix_console_scripts(stage: Path) -> None:
    """Windows で集めると bin/ に .exe ができるので消し、エントリポイントに使う opentelemetry-instrument だけ書く。"""
    bin_dir = stage / "bin"
    if bin_dir.exists():
        for p in bin_dir.glob("*.exe"):
            p.unlink()
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "opentelemetry-instrument").write_text(OTEL_SCRIPT, encoding="utf-8", newline="\n")


def write_zip(src: Path, out: Path) -> tuple[int, int]:
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if "__pycache__" in p.parts:
                continue
            arc = p.relative_to(src).as_posix()
            if p.is_dir():
                info = zipfile.ZipInfo(arc + "/")
                info.external_attr = (0o40755 << 16) | 0x10
                z.writestr(info, b"")
                continue
            info = zipfile.ZipInfo.from_file(p, arc)
            mode = 0o755 if p.parent.name == "bin" else 0o644
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            data = p.read_bytes()
            total += len(data)
            z.writestr(info, data)
    return out.stat().st_size, total


def build_agent(name: str, files: list[str]) -> Path:
    stage = BUILD / name
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    install_deps(ROOT / "agents" / name / "requirements.txt", stage)
    fix_console_scripts(stage)
    for f in files:
        shutil.copy2(ROOT / "agents" / name / f, stage / f)
    copy_shared(stage)
    out = DIST / f"{name}.zip"
    zipped, unzipped = write_zip(stage, out)
    print(f"{out.name}: {zipped / 1e6:.1f} MB (unzipped {unzipped / 1e6:.1f} MB)")
    if zipped > ZIP_LIMIT or unzipped > UNZIPPED_LIMIT:
        sys.exit(f"{out.name} exceeds the AgentCore limit (250 MB zipped / 750 MB unzipped)")
    return out


def build_lambda() -> Path:
    stage = DIST / "lambda_analytics"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy2(ROOT / "lambdas" / "analytics_tools" / "handler.py", stage / "handler.py")
    copy_shared(stage, with_vlmlab=False)  # boto3 は Lambda の実行環境のものを使う
    return stage


def main() -> None:
    DIST.mkdir(exist_ok=True)
    for name, files in AGENTS.items():
        build_agent(name, files)
    print("lambda:", build_lambda())


if __name__ == "__main__":
    main()
