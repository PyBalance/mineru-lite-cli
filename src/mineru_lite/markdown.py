from __future__ import annotations

from typing import List
from PIL import Image, ImageDraw

from .model import ContentBlock
from .tables import html_table_filter


def page_to_markdown(page_img: Image.Image, blocks: List[ContentBlock]) -> str:
    md_parts: list[str] = []

    for b in blocks:
        if b.type == "__deleted_table__":
            continue
        elif b.type == "text":
            text = (b.content or "").strip()
            if text:
                md_parts.append(text)
        elif b.type == "equation":
            tex = (b.content or "").strip()
            if tex:
                md_parts.append(f"$\n{tex}\n$")
        elif b.type == "table":
            html = html_table_filter(b.content or "")
            if html:
                md_parts.append("\n" + html + "\n")
        elif b.type == "image":
            md_parts.append("（图片略）")
        else:
            if b.content:
                md_parts.append(b.content)

    return "\n\n".join(md_parts).strip() + "\n"


def render_debug_pdf(pages_img: List[Image.Image], pages_blocks: List[List[ContentBlock]]) -> bytes:
    vis_pages: list[Image.Image] = []
    for img, blocks in zip(pages_img, pages_blocks):
        vis = img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis)
        W, H = vis.size
        for b in blocks:
            x0, y0, x1, y1 = b.bbox
            box = (x0 * W, y0 * H, x1 * W, y1 * H)
            draw.rectangle(box, outline=(0, 128, 255), width=3)
        vis_pages.append(vis)

    import io

    buf = io.BytesIO()
    if vis_pages:
        first, rest = vis_pages[0], vis_pages[1:]
        first.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()