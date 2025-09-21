from __future__ import annotations

import os
from typing import List, Optional

from PIL import Image

# pypdfium2 兼容导入
try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover
    pdfium = None


def ensure_pdfium() -> None:
    if pdfium is None:
        raise RuntimeError("需要 pypdfium2：pip install pypdfium2")


def parse_pages_spec(spec: Optional[str], total_pages: int) -> List[int]:
    if not spec or not spec.strip():
        return list(range(total_pages))

    spec = spec.replace(" ", "")
    result: set[int] = set()
    tokens = [t for t in spec.split(",") if t]

    for tok in tokens:
        if "-" in tok:
            left, right = tok.split("-", 1)
            l = int(left) if left.isdigit() else 1
            r = int(right) if right.isdigit() else total_pages
        else:
            if not tok.isdigit():
                continue
            l = r = int(tok)

        l = max(1, min(l, total_pages))
        r = max(1, min(r, total_pages))

        if l <= r:
            for p in range(l, r + 1):
                result.add(p - 1)

    return sorted(result)


def _render_page_to_pil(page, scale: float) -> Image.Image:
    try:
        pil = page.render(scale=scale).to_pil()
    except AttributeError:
        pil = page.render_to(pdfium.BitmapConv.pil_image, scale=scale)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 220, page_indices: Optional[List[int]] = None) -> List[Image.Image]:
    ensure_pdfium()
    pdf = pdfium.PdfDocument(pdf_bytes)
    total = len(pdf)
    indices = range(total) if page_indices is None else page_indices
    images: List[Image.Image] = []
    scale = dpi / 72.0
    for i in indices:
        page = pdf[i]
        pil = _render_page_to_pil(page, scale)
        images.append(pil)
    return images


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    i = 0
    while s >= 1024 and i < len(units) - 1:
        s /= 1024.0
        i += 1
    out = f"{s:.2f}".rstrip("0").rstrip(".")
    return f"{out} {units[i]}"


def _normalize_text(t: str) -> str:
    import re

    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def summarize_pdf(path: str, pages_spec: Optional[str]) -> None:
    ensure_pdfium()
    if not path.lower().endswith(".pdf"):
        raise RuntimeError("summary 仅支持 PDF 文件。")

    file_size = os.path.getsize(path)
    pdf = pdfium.PdfDocument(path)
    total_pages = len(pdf)

    wanted = parse_pages_spec(pages_spec, total_pages)
    total_chars = 0

    print(f"{os.path.basename(path)} 的 summary")
    print(f"文件大小: {_human_bytes(file_size)}")
    print(f"总页数: {total_pages}")

    per_page_info = []
    for i in wanted:
        page = pdf[i]
        width_pt, height_pt = page.get_size()
        mm_per_pt = 25.4 / 72.0
        w_mm = width_pt * mm_per_pt
        h_mm = height_pt * mm_per_pt
        orient = "纵向" if height_pt >= width_pt else "横向"

        text = ""
        page_chars = 0
        try:
            tp = page.get_textpage()
            try:
                n = tp.count_chars()
                page_chars = int(n)
                text = tp.get_text_range(0, n) if n > 0 else ""
            except Exception:
                try:
                    text = tp.get_text_bounded(0, 0, width_pt, height_pt)
                    page_chars = len(text)
                except Exception:
                    text = ""
                    page_chars = 0
        except Exception:
            text = ""
            page_chars = 0

        total_chars += page_chars
        per_page_info.append(
            {
                "index": i + 1,
                "w_pt": width_pt,
                "h_pt": height_pt,
                "w_mm": w_mm,
                "h_mm": h_mm,
                "orient": orient,
                "chars": page_chars,
                "text": text,
            }
        )

    print(f"总字符数（文字层，含空白）: {total_chars}")
    print()

    for info in per_page_info:
        idx = info["index"]
        wpt, hpt = info["w_pt"], info["h_pt"]
        wmm, hmm = info["w_mm"], info["h_mm"]
        orient = info["orient"]
        chars = info["chars"]
        text = info["text"]

        print(f"=== Page {idx} ===")
        print(f"尺寸: {int(wpt)} x {int(hpt)} pt (≈ {wmm:.1f} x {hmm:.1f} mm), {orient}")
        print(f"文字层字数: {chars}")

        norm = _normalize_text(text)
        if not norm:
            print("无文字层(但可能包含扫描件)")
            print()
            continue

        head = norm[:100]
        tail = norm[-30:] if len(norm) > 30 else norm
        print("文字预览：")
        print("开头[最多100字]:")
        print(head)
        print()
        print("结尾[最多30字]:")
        print(tail)
        print()