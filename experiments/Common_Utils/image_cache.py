# Experiment/experiments/Common_Utils/image_cache.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from urllib.request import urlretrieve

DEBUG = os.environ.get("QWEN_IMAGE_CACHE_DEBUG", "0") == "1"

_DEFAULT_CACHE_DIR = (Path(__file__).resolve().parents[1] / "image_cache").resolve()

CACHE_DIR = Path(os.environ.get("QWEN_IMAGE_CACHE", str(_DEFAULT_CACHE_DIR))).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALLOW_DOWNLOAD = os.environ.get("QWEN_IMAGE_CACHE_DOWNLOAD", "1") == "1"

def url_to_cached_path(url: str) -> str:
    """
    1) If CACHE_DIR/<basename> exists, use the local path.
    2) If not:
    - If QWEN_IMAGE_CACHE_DOWNLOAD=1, download it and then use the local path.
    - Otherwise, return the original URL as is.
    """
    if not url or not isinstance(url, str):
        return url

    if url.startswith("http"):
        base = url.split("?")[0].split("/")[-1]
        if not base:
            return url
        local = CACHE_DIR / base

        if local.exists() and local.stat().st_size > 0:
            if DEBUG:
                print(f"[cache hit] {url} -> {local}")
            return str(local)

        if ALLOW_DOWNLOAD:
            try:
                urlretrieve(url, local)
                if local.exists() and local.stat().st_size > 0:
                    if DEBUG:
                        print(f"[cache dl] {url} -> {local}")
                    return str(local)
            except Exception:
                try:
                    if local.exists():
                        local.unlink()
                except Exception:
                    pass
        if DEBUG:
            print(f"[cache miss] {url} -> (no local file)")
        return url

    return url
