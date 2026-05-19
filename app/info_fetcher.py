import json
import time
from pathlib import Path
from typing import Optional

import requests

CACHE_DIR = Path("cache")
USER_AGENT = "OpenCV-Object-Detector/1.0 (educational; contact: user@example.com)"
TIMEOUT = 6


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower())


def _cache_path(label: str, lang: str) -> Path:
    return CACHE_DIR / f"{lang}_{_safe(label)}.json"


def _read_cache(label: str, lang: str) -> Optional[dict]:
    p = _cache_path(label, lang)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(label: str, lang: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(label, lang).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _resolve_title(label: str, lang: str) -> Optional[str]:
    """Pick the top opensearch hit for `label` on the given Wikipedia. None if no hit."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": label,
        "limit": 1,
        "namespace": 0,
        "format": "json",
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and len(data) >= 2 and data[1]:
        return data[1][0]
    return None


def _fetch_summary(title: str, lang: str) -> dict:
    encoded = requests.utils.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_details(label: str, *, lang: str = "en", force_refresh: bool = False) -> dict:
    cached = None if force_refresh else _read_cache(label, lang)
    if cached:
        return cached

    try:
        title = _resolve_title(label, lang)
        # Fall back to English Wikipedia if the target language has no match for the
        # English class label. The YOLO labels are always English, so this is common
        # for languages like Hindi/Telugu when the article uses a localized name.
        fallback_lang = lang
        if title is None and lang != "en":
            title = _resolve_title(label, "en")
            fallback_lang = "en"
        if title is None:
            title = label

        payload = _fetch_summary(title, fallback_lang)
        details = {
            "label": label,
            "lang": fallback_lang,
            "title": payload.get("title", title),
            "summary": payload.get("extract", ""),
            "url": payload.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "image_url": payload.get("thumbnail", {}).get("source", ""),
            "fetched_at": time.time(),
            "source": f"wikipedia ({fallback_lang})",
        }
        _write_cache(label, lang, details)
        return details
    except requests.RequestException as e:
        return {
            "label": label,
            "lang": lang,
            "title": label.title(),
            "summary": f"(offline) Could not reach Wikipedia: {e}. Connect to the internet to fetch details.",
            "url": "",
            "image_url": "",
            "fetched_at": time.time(),
            "source": "offline",
        }
