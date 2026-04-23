"""
backend/i18n_messages.py — Centralized localized API messages.

Keep API-facing status/error strings here so routers can return localized responses
based on request language.
"""

from __future__ import annotations

from i18n import Lang

_MESSAGES: dict[str, dict[Lang, str]] = {
    "model_not_ready": {
        "en": "Server is still starting up. Please try again in a few seconds.",
        "vi": "Server đang khởi động. Vui lòng thử lại sau vài giây.",
    },
    "data_not_loaded": {
        "en": "Data not yet loaded.",
        "vi": "Dữ liệu chưa được tải.",
    },
    "prediction_error": {
        "en": "Prediction failed due to an internal error.",
        "vi": "Dự đoán thất bại do lỗi nội bộ.",
    },
    "history_not_ready": {
        "en": "History storage is not initialized.",
        "vi": "Lưu trữ lịch sử chưa được khởi tạo.",
    },
    "history_save_failed": {
        "en": "Could not save analysis history.",
        "vi": "Không thể lưu lịch sử phân tích.",
    },
}


def msg(key: str, lang: str = "en") -> str:
    """Return localized message for `key`, defaulting to English."""
    normalized: Lang = "vi" if str(lang).strip().lower().startswith("vi") else "en"
    bucket = _MESSAGES.get(key)
    if not bucket:
        return key
    return bucket.get(normalized) or bucket["en"]

