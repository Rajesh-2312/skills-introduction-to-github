import json
import time
from pathlib import Path
from typing import Optional

import requests

CACHE_DIR = Path("cache")
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "OpenCV-Object-Detector/1.0 (educational; contact: user@example.com)"
TIMEOUT = 6


def _cache_path(label: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in label.lower())
    return CACHE_DIR / f"{safe}.json"


def _read_cache(label: str) -> Optional[dict]:
    p = _cache_path(label)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(label: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(label).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_title(label: str) -> str:
    # Disambiguates labels like "mouse" (animal vs computer mouse) by picking
    # the top opensearch hit before requesting the summary.
    params = {
        "action": "opensearch",
        "search": label,
        "limit": 1,
        "namespace": 0,
        "format": "json",
    }
    r = requests.get(
        WIKI_SEARCH_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and len(data) >= 2 and data[1]:
        return data[1][0]
    return label


def fetch_details(label: str, *, force_refresh: bool = False) -> dict:
    cached = None if force_refresh else _read_cache(label)
    if cached:
        return cached

    try:
        title = _resolve_title(label)
        url = WIKI_SUMMARY_URL.format(title=requests.utils.quote(title.replace(" ", "_")))
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        details = {
            "label": label,
            "title": payload.get("title", title),
            "summary": payload.get("extract", ""),
            "url": payload.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "image_url": payload.get("thumbnail", {}).get("source", ""),
            "fetched_at": time.time(),
            "source": "wikipedia",
        }
        _write_cache(label, details)
        return details
    except requests.RequestException as e:
        return {
            "label": label,
            "title": label.title(),
            "summary": f"(offline) Could not reach Wikipedia: {e}. Connect to the internet to fetch details.",
            "url": "",
            "image_url": "",
            "fetched_at": time.time(),
            "source": "offline",
        }
