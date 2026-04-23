"""
backend/i18n.py — Request language resolution and helpers for localized API responses.

This module implements a small, backend-driven i18n layer:
- Resolve request language from `?lang=` query param or `Accept-Language` header
- Provide a helper to pick language-specific fields from legacy dict payloads
"""

from __future__ import annotations

from typing import Literal

from fastapi import Header, Query

Lang = Literal["en", "vi"]


def _normalize_lang(raw: str | None) -> Lang | None:
    if not raw:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value.startswith("vi"):
        return "vi"
    if value.startswith("en"):
        return "en"
    return None


def resolve_lang(
    lang: str | None = Query(default=None),
    accept_language: str | None = Header(default=None, alias="Accept-Language"),
) -> Lang:
    """
    FastAPI dependency that resolves request language.

    Priority:
      1) `?lang=` query param
      2) `Accept-Language` HTTP header
      3) default to "en"
    """
    query_lang = _normalize_lang(lang)
    if query_lang:
        return query_lang

    if accept_language:
        # Example: "vi-VN,vi;q=0.9,en;q=0.8" -> "vi-VN" -> "vi"
        first = accept_language.split(",", 1)[0].split(";", 1)[0]
        header_lang = _normalize_lang(first)
        if header_lang:
            return header_lang

    return "en"


def pick(data: dict, key: str, lang: Lang) -> str:
    """
    Read `f"{key}_{lang}"` from `data` with fallback to `f"{key}_en"`.
    Returns empty string if the key is missing.
    """
    value = data.get(f"{key}_{lang}")
    if value is None:
        value = data.get(f"{key}_en")
    return "" if value is None else str(value)

