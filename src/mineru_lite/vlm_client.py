from __future__ import annotations

from typing import List
from PIL import Image

from .model import ContentBlock

try:
    from mineru_vl_utils import MinerUClient
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Please install mineru-vl-utils: pip install mineru-vl-utils==0.1.8\n"
        f"ImportError: {e}"
    )


class VLMHttpClient:
    """HTTP client for VLM service."""

    def __init__(self, server_url: str):
        self.client = MinerUClient(backend="http-client", server_url=server_url)

    def infer_one_image(self, image: Image.Image) -> List[ContentBlock]:
        blocks = self.client.two_step_extract(image)
        return [ContentBlock.from_any(b) for b in blocks]