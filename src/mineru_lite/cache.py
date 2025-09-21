from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from platformdirs import PlatformDirs

from .config import APP_NAME


def cache_dir() -> Path:
    d = PlatformDirs(appname=APP_NAME, appauthor=False)
    p = Path(d.user_cache_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_path_for_hash(file_hash: str) -> Path:
    return cache_dir() / f"{file_hash}.json"


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def load_pdf_cache(file_hash: str) -> Dict[str, Any]:
    fp = cache_path_for_hash(file_hash)
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text("utf-8"))
    except Exception:
        return {}


def save_pdf_cache_atomic(file_hash: str, obj: Dict[str, Any]) -> None:
    fp = cache_path_for_hash(file_hash)
    tmp = Path(
        tempfile.mkstemp(prefix="cache-", suffix=".json", dir=str(fp.parent))[1]
    )
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        tmp.replace(fp)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass