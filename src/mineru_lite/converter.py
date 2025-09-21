from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from PIL import Image

from .blocks import sort_blocks_reading_order
from .cache import load_pdf_cache, save_pdf_cache_atomic, sha256_bytes
from .markdown import page_to_markdown, render_debug_pdf
from .model import ContentBlock
from .pdf_utils import ensure_pdfium, parse_pages_spec
from .tables import merge_cross_page_tables
from .vlm_client import VLMHttpClient

# pypdfium2 仅在 PDF 模式用到
try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover
    pdfium = None


class VLMHttpToMarkdown:
    def __init__(self, server_url: str):
        self.client = VLMHttpClient(server_url)

    def _extract_page(self, img: Image.Image) -> List[ContentBlock]:
        blocks = self.client.infer_one_image(img)
        return sort_blocks_reading_order(blocks)

    def from_images(self, images: List[Image.Image], *, make_debug_pdf: bool = False) -> Dict[str, Any]:
        pages_img = images
        pages_blocks = [self._extract_page(im) for im in pages_img]
        merge_cross_page_tables(pages_blocks)

        md_pages = [page_to_markdown(img, blks) for img, blks in zip(pages_img, pages_blocks)]
        md = "\n".join(md_pages)

        middle = {
            "pdf_info": {
                "page_count": len(pages_img),
                "pages": [
                    {
                        "width": img.size[0],
                        "height": img.size[1],
                        "blocks": [b.__dict__ for b in blks],
                    }
                    for img, blks in zip(pages_img, pages_blocks)
                ],
            }
        }

        result: Dict[str, Any] = {"markdown": md, "middle_json": middle}
        if make_debug_pdf:
            result["debug_pdf_bytes"] = render_debug_pdf(pages_img, pages_blocks)
        return result

    def from_pdf_bytes(
        self,
        pdf_bytes: bytes,
        *,
        dpi: int = 220,
        make_debug_pdf: bool = False,
        pages_spec: Optional[str] = None,
    ) -> Dict[str, Any]:
        ensure_pdfium()
        assert pdfium is not None

        file_hash = sha256_bytes(pdf_bytes)
        pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
        total_pages = len(pdf)
        wanted = parse_pages_spec(pages_spec, total_pages)

        cache = load_pdf_cache(file_hash)
        if not cache or cache.get("dpi") != dpi:
            cache = {
                "version": 1,
                "file_sha256": file_hash,
                "page_count": total_pages,
                "dpi": dpi,
                "pages": {},
            }
        pages_cache: dict = cache.setdefault("pages", {})

        # 渲染+推理缺失页，并写入缓存
        missing = [i for i in wanted if str(i) not in pages_cache]
        if missing:
            scale = dpi / 72.0
            for i in missing:
                page = pdf[i]
                try:
                    pil = page.render(scale=scale).to_pil()
                except AttributeError:
                    pil = page.render_to(pdfium.BitmapConv.pil_image, scale=scale)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")

                blocks = self._extract_page(pil)
                pages_cache[str(i)] = {
                    "width": pil.size[0],
                    "height": pil.size[1],
                    "blocks": [b.__dict__ for b in blocks],
                }
            save_pdf_cache_atomic(file_hash, cache)

        # 组装本次运行需要的页图与块
        pages_img: List[Image.Image] = []
        pages_blocks: List[List[ContentBlock]] = []
        scale = dpi / 72.0
        for i in wanted:
            meta = pages_cache[str(i)]
            page = pdf[i]
            try:
                pil = page.render(scale=scale).to_pil()
            except AttributeError:
                pil = page.render_to(pdfium.BitmapConv.pil_image, scale=scale)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            pages_img.append(pil)
            blocks = [ContentBlock.from_any(b) for b in meta["blocks"]]
            pages_blocks.append(blocks)

        merge_cross_page_tables(pages_blocks)

        md_pages = [page_to_markdown(img, blks) for img, blks in zip(pages_img, pages_blocks)]
        md = "\n".join(md_pages)

        middle = {
            "pdf_info": {
                "page_count": len(pages_img),
                "pages": [
                    {
                        "width": img.size[0],
                        "height": img.size[1],
                        "blocks": [b.__dict__ for b in blks],
                    }
                    for img, blks in zip(pages_img, pages_blocks)
                ],
            }
        }

        result: Dict[str, Any] = {"markdown": md, "middle_json": middle}
        if make_debug_pdf:
            result["debug_pdf_bytes"] = render_debug_pdf(pages_img, pages_blocks)
        return result


def build_markdown_from_pdf_bytes(
    pdf_bytes: bytes,
    *,
    dpi: int = 220,
    make_debug_pdf: bool = False,
    server_url: str,
    pages_spec: Optional[str] = None,
) -> Dict[str, Any]:
    return VLMHttpToMarkdown(server_url=server_url).from_pdf_bytes(
        pdf_bytes, dpi=dpi, make_debug_pdf=make_debug_pdf, pages_spec=pages_spec
    )


def build_markdown_from_images(
    images: List[Image.Image], *, make_debug_pdf: bool = False, server_url: str
) -> Dict[str, Any]:
    return VLMHttpToMarkdown(server_url=server_url).from_images(
        images, make_debug_pdf=make_debug_pdf
    )