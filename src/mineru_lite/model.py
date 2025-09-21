from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class ContentBlock:
    type: str
    bbox: Tuple[float, float, float, float]
    angle: Optional[int]
    content: Optional[str]

    @staticmethod
    def from_any(x: Any) -> "ContentBlock":
        t = getattr(x, "type", None) or x.get("type")
        b = getattr(x, "bbox", None) or x.get("bbox")
        a = getattr(x, "angle", None) if hasattr(x, "angle") else x.get("angle")
        c = (
            getattr(x, "content", None)
            if hasattr(x, "content")
            else x.get("content")
        )
        return ContentBlock(type=t, bbox=tuple(b), angle=a, content=c)

    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def mid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)