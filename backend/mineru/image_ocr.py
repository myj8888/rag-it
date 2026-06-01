"""Image OCR helper.

Input: local image files extracted by MinerU.
Output: OCR text for indexing.
Side effects: loads RapidOCR model locally.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_image(image_path: Path) -> str:
    if not image_path.exists():
        logger.warning("Image for OCR does not exist: %s", image_path)
        return ""
    logger.info("Running OCR for image: %s", image_path)
    result, _ = _ocr_engine()(str(image_path))
    if not result:
        return ""
    lines = [str(item[1]).strip() for item in result if len(item) > 1 and str(item[1]).strip()]
    return "\n".join(lines)

