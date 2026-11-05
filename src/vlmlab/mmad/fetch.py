"""Hugging Face の ZIP から、使う画像だけを HTTP の範囲指定で取り出す。

ZIP は全部で約28 GB あるが、1,050問で使う画像は約2,000枚なので、目次と必要なファイルだけを読む。
"""

from __future__ import annotations

import io
import threading
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import HF_REVISION, HF_ZIP_URL


class HttpRangeFile(io.RawIOBase):
    """読み取り専用・シークできるファイルとして、URL を範囲指定で読む。"""

    def __init__(self, url: str, timeout: float = 120.0):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.url = r.geturl()  # リダイレクト先（CDN）を覚えて、以降は直接読む
            self.size = int(r.headers["Content-Length"])
        self.timeout = timeout
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[override]
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise IOError(f"server ignored Range request (status {r.status})")
            data = r.read()
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


def _open_zip(source: str, revision: str) -> zipfile.ZipFile:
    url = HF_ZIP_URL.format(rev=revision, source=source)
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


def fetch_images(keys: list[str], images_root: str | Path, revision: str = HF_REVISION, workers: int = 8) -> dict[str, int]:
    """keys（"VisA/candle/..." の形）を images_root の下へ取り出す。既にあるファイルは取らない。

    ZIP の中のパスは、キーと同じ（先頭が元のデータセット名）。見つからないキーがあれば KeyError。
    戻り値は {"fetched": 取った数, "skipped": 既にあった数}。
    """
    root = Path(images_root)
    todo: dict[str, list[str]] = defaultdict(list)
    skipped = 0
    for key in keys:
        if (root / key).exists():
            skipped += 1
        else:
            todo[key.split("/")[0]].append(key)

    fetched = 0
    lock = threading.Lock()
    for source, source_keys in sorted(todo.items()):
        names = set(_open_zip(source, revision).namelist())
        missing = [k for k in source_keys if k not in names]
        if missing:
            raise KeyError(f"{len(missing)} keys not in {source}.zip, e.g. {missing[:3]}")
        local = threading.local()

        def get(key: str) -> None:
            nonlocal fetched
            if not hasattr(local, "zf"):
                local.zf = _open_zip(source, revision)  # ZipFile はスレッドごとに開く
            data = local.zf.read(key)
            dest = root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            part.write_bytes(data)
            part.replace(dest)
            with lock:
                fetched += 1

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(get, source_keys))
    return {"fetched": fetched, "skipped": skipped}
