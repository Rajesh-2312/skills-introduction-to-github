from functools import lru_cache

import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


@lru_cache(maxsize=1)
def is_available() -> bool:
    """Check that both pytesseract and the tesseract binary are usable."""
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text(image: np.ndarray, lang: str = "eng") -> str:
    """OCR a BGR image crop. Returns '' if tesseract isn't installed or fails."""
    if not is_available() or image is None or image.size == 0:
        return ""
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return " ".join(text.split())  # collapse whitespace/newlines
    except Exception as e:
        print(f"[ocr] error: {e}")
        return ""
