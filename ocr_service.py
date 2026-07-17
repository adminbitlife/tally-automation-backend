from __future__ import annotations

import os
import re
from typing import List

import pytesseract
from PIL import Image
from paddleocr import PaddleOCR


class OCRService:
    def __init__(self) -> None:
        self._ocr: PaddleOCR | None = None

    def _get_ocr(self) -> PaddleOCR:
        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
                det=False,
                rec=True,
                cls=False,
            )
        return self._ocr

    def _clean_text_lines(self, lines: List[str]) -> List[str]:
        cleaned: List[str] = []
        for line in lines:
            if not isinstance(line, str):
                continue
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"\s+", " ", line)
            cleaned.append(line)
        return cleaned

    def _extract_with_tesseract(self, image_path: str) -> List[str]:
        if not os.path.exists(image_path):
            return []

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return self._clean_text_lines(text.splitlines())

    def extract_text(self, image_path: str) -> List[str]:
        try:
            ocr = self._get_ocr()
            result = ocr.ocr(image_path, cls=True)
            if not result:
                return self._extract_with_tesseract(image_path)

            texts: List[str] = []
            if isinstance(result[0], list):
                for item in result[0]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                        if isinstance(text, str) and text.strip():
                            texts.append(text)

            if texts:
                return self._clean_text_lines(texts)
        except Exception:
            pass

        return self._extract_with_tesseract(image_path)


ocr_service = OCRService()
