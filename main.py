"""
VLM http-client → Markdown（全内存实现）
=================================================
- 仅使用 `mineru-vl-utils` 的 http-client（服务端固定： http://192.168.191.126:30000）。
- 支持输入：PDF（二进制/bytes）或 PIL.Image（单张/多张）。
- 流程：PDF渲图 → 每页走 VLM two_step_extract → 页内阅读顺序整理 → 跨页表格合并 → 产出 Markdown。
- 全部在内存中运算；可选返回：middle_json-like 结构、调试用“拼接PDF”（叠加可视化框）。

依赖（请在你的环境安装）：
    pip install mineru-vl-utils==0.1.8 pypdfium2 pillow beautifulsoup4 numpy

注意：mineru-vl-utils 只接受“图像输入”，因此 PDF 会被先渲成图片；
mineru-vl-utils 返回的 `extracted_blocks` 为每页的布局块列表（text/table/equation/image）。
本实现参考 MinerU 主仓 `backend/vlm` 后处理思想：
- 表格跨页合并（通过 HTML 解析 + 结构/列数一致性判断）；
- 阅读顺序整理（启发式：按列 → 行排序）；
- 表格 HTML 符号反转义（`<` `>` `≤` `≥`等）。

作者备注：该实现为“轻量复刻”，接口与细节与 MinerU 源码不同，但达成等价功能目标。
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
import numpy as np
from bs4 import BeautifulSoup

try:
    import pypdfium2 as pdfium
except Exception as e:  # 允许在仅图片输入场景无 pypdfium2
    pdfium = None

try:
    from mineru_vl_utils import MinerUClient
except Exception as e:
    raise RuntimeError(
        "请先安装 mineru-vl-utils： pip install mineru-vl-utils==0.1.8\n"
        f"ImportError: {e}"
    )


# ------------------------------
# 常量 & 工具
# ------------------------------
VLM_SERVER_URL = "http://192.168.191.126:30000"  # 固定为用户提供的 URL


@dataclass
class ContentBlock:
    """统一 extracted_blocks 的访问，兼容 dict / obj 两种形态。"""
    type: str
    bbox: Tuple[float, float, float, float]  # [xmin, ymin, xmax, ymax], 0~1 归一化
    angle: Optional[int]
    content: Optional[str]

    @staticmethod
    def from_any(x: Any) -> "ContentBlock":
        # 兼容 dict 或具名对象（README 里称 ContentBlock 对象）
        t = getattr(x, "type", None) or x.get("type")
        b = getattr(x, "bbox", None) or x.get("bbox")
        a = getattr(x, "angle", None) if hasattr(x, "angle") else x.get("angle")
        c = getattr(x, "content", None) if hasattr(x, "content") else x.get("content")
        return ContentBlock(type=t, bbox=tuple(b), angle=a, content=c)

    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def mid(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# ------------------------------
# PDF → 图像（内存）
# ------------------------------

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 220) -> List[Image.Image]:
    """将 PDF 渲为 PIL 图片列表（RGB），不落盘。兼容不同版本 pypdfium2。"""
    if pdfium is None:
        raise RuntimeError("未安装 pypdfium2，无法渲 PDF；请 pip install pypdfium2")

    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    images: List[Image.Image] = []
    scale = dpi / 72.0  # 72pt/inch

    for i in range(len(pdf)):
        page = pdf[i]
        # 新版 pypdfium2：page.render(...).to_pil()
        try:
            pil = page.render(scale=scale).to_pil()
        except AttributeError:
            # 旧版 fallback：page.render_to(...)
            pil = page.render_to(pdfium.BitmapConv.pil_image, scale=scale)

        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        images.append(pil)

    return images

# ------------------------------
# VLM 客户端（http-client）
# ------------------------------

class VLMHttpClient:
    def __init__(self, server_url: str = VLM_SERVER_URL):
        self.client = MinerUClient(backend="http-client", server_url=server_url)

    def infer_one_image(self, image: Image.Image) -> List[ContentBlock]:
        blocks = self.client.two_step_extract(image)
        return [ContentBlock.from_any(b) for b in blocks]


# ------------------------------
# 阅读顺序排序（页内）
# ------------------------------

def sort_blocks_reading_order(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """简化版阅读顺序：
    - 先按列（x 中值聚类为 1~3 列，启发式）
    - 列内按上→下、同排按左→右
    """
    if not blocks:
        return blocks

    mids = np.array([b.mid()[0] for b in blocks])  # 仅用 x
    mids_sorted_idx = np.argsort(mids)
    mids_sorted = mids[mids_sorted_idx]

    # 列切分：看 x 间隙是否存在“明显断点”（>0.15）
    gaps = np.diff(mids_sorted)
    cut_points = np.where(gaps > 0.15)[0]

    # 最多切成 3 列
    if len(cut_points) == 0:
        columns = [blocks]
    else:
        # 根据 cut_points 将 blocks 切片
        parts = np.split(mids_sorted_idx, cut_points + 1)
        columns = [[blocks[i] for i in part] for part in parts]

    def sort_in_column(col: List[ContentBlock]) -> List[ContentBlock]:
        # 先按 y0，再按 x0
        return sorted(col, key=lambda b: (b.bbox[1], b.bbox[0]))

    columns = [sort_in_column(col) for col in columns]

    # 列顺序：按列里元素的平均 x0 升序
    col_with_key = []
    for col in columns:
        if col:
            x0_mean = float(np.mean([b.bbox[0] for b in col]))
        else:
            x0_mean = 0.0
        col_with_key.append((x0_mean, col))
    col_with_key.sort(key=lambda t: t[0])

    ordered = []
    for _, col in col_with_key:
        ordered.extend(col)
    return ordered


# ------------------------------
# 表格处理（HTML 解析 & 反转义 & 跨页合并）
# ------------------------------

_HTML_UNESCAPE_MAP = {
    "&amp;lt;": "<",
    "&amp;gt;": ">",
    "&amp;le;": "≤",
    "&amp;ge;": "≥",
}

def html_table_filter(html: str) -> str:
    for k, v in _HTML_UNESCAPE_MAP.items():
        html = html.replace(k, v)
    return html


def _parse_table(html: str) -> Tuple[List[List[str]], List[List[str]]]:
    """解析 HTML 表格，返回 (thead_rows, tbody_rows)。
    只提取 <th>/<td> 的纯文本（保留最主要结构用于合并判定）。
    """
    soup = BeautifulSoup(html, "html.parser")
    thead_rows, tbody_rows = [], []
    thead = soup.find("thead")
    if thead:
        for tr in thead.find_all("tr"):
            thead_rows.append([c.get_text(strip=True) for c in tr.find_all(["th", "td"])])
    tbody = soup.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            tbody_rows.append([c.get_text(strip=True) for c in tr.find_all(["th", "td"])])
    else:
        # 没有 tbody 就直接找所有行
        for tr in soup.find_all("tr"):
            tbody_rows.append([c.get_text(strip=True) for c in tr.find_all(["th", "td"])])
    return thead_rows, tbody_rows


def _count_columns(rows: List[List[str]]) -> int:
    return int(np.median([len(r) for r in rows])) if rows else 0


def _rebuild_table_html(thead_rows: List[List[str]], tbody_rows: List[List[str]]) -> str:
    """根据行数据重建简单 HTML 表格（保留 thead / tbody）。"""
    def tr_html(cells: List[str], is_header: bool) -> str:
        tag = "th" if is_header else "td"
        return "<tr>" + "".join(f"<{tag}>{BeautifulSoup(c, 'html.parser').get_text()}</{tag}>" for c in cells) + "</tr>"

    parts = ["<table>"]
    if thead_rows:
        parts.append("<thead>")
        parts.extend(tr_html(r, True) for r in thead_rows)
        parts.append("</thead>")
    parts.append("<tbody>")
    parts.extend(tr_html(r, False) for r in tbody_rows)
    parts.append("</tbody></table>")
    return "".join(parts)


def merge_cross_page_tables(pages: List[List[ContentBlock]]) -> None:
    """就地跨页表格合并：
    - 若前一页末尾、后一页开头都是 table，且列数匹配，则把后一页 tbody 追加到前一页。
    - 合并后，后一页对应表格块标记为删除（type 置为 '__deleted_table__'）。
    """
    for p in range(1, len(pages)):
        prev_blocks = [b for b in pages[p - 1] if b.type == "table"]
        curr_blocks = [b for b in pages[p] if b.type == "table"]
        if not prev_blocks or not curr_blocks:
            continue
        prev = prev_blocks[-1]  # 取上一页最后一个表格
        curr = curr_blocks[0]   # 取下一页第一个表格
        if not prev.content or not curr.content:
            continue

        prev_html = html_table_filter(prev.content)
        curr_html = html_table_filter(curr.content)
        p_thead, p_tbody = _parse_table(prev_html)
        c_thead, c_tbody = _parse_table(curr_html)

        # 列数匹配 + thead 文本相似（若都存在）
        p_cols = _count_columns(p_thead or p_tbody)
        c_cols = _count_columns(c_thead or c_tbody)
        if p_cols == 0 or c_cols == 0 or abs(p_cols - c_cols) > 0:
            continue

        def rows_text(rows: List[List[str]]) -> List[str]:
            return ["|".join(r) for r in rows]

        # 如果两页 thead 完全相同，则认为第二页 thead 可能是重复表头，合并时仅保留第一页 thead
        if p_thead and c_thead and rows_text(p_thead) == rows_text(c_thead):
            new_thead = p_thead
        else:
            # 否则保留前一页 thead；若前一页没有而后一页有，就并上
            new_thead = p_thead or c_thead

        new_tbody = p_tbody + c_tbody
        merged_html = _rebuild_table_html(new_thead, new_tbody)

        prev.content = merged_html
        curr.type = "__deleted_table__"  # 标记删除


# ------------------------------
# Markdown 生成（页内图像裁剪内嵌 / 文本 / 公式 / 表格）
# ------------------------------

def crop_block_image_to_data_uri(page_image: Image.Image, block: ContentBlock, jpeg: bool = False) -> str:
    x0, y0, x1, y1 = block.bbox
    W, H = page_image.size
    box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    crop = page_image.crop(box)
    buf = io.BytesIO()
    if jpeg:
        crop.convert("RGB").save(buf, format="JPEG", quality=88)
        mime = "image/jpeg"
    else:
        crop.save(buf, format="PNG")
        mime = "image/png"
    b64 = buf.getvalue()
    import base64
    return f"data:{mime};base64,{base64.b64encode(b64).decode()}"


def page_to_markdown(page_img: Image.Image, blocks: List[ContentBlock]) -> str:
    md_parts: List[str] = []
    for b in blocks:
        if b.type == "__deleted_table__":
            continue
        if b.type == "text":
            text = (b.content or "").strip()
            if text:
                md_parts.append(text)
        elif b.type == "equation":
            tex = (b.content or "").strip()
            if tex:
                md_parts.append(f"$$\n{tex}\n$$")
        elif b.type == "table":
            html = html_table_filter(b.content or "")
            if html:
                # 直接嵌入 HTML 表格（GitHub/大多 Markdown 渲染器都支持）
                md_parts.append("\n" + html + "\n")
        elif b.type == "image":
            # 仅输出占位说明，不内联 base64
            md_parts.append("（图片略）")
        else:
            # 其他类型（如 list/title/toc 等），按文本兜底
            if b.content:
                md_parts.append(b.content)
    # 块之间空行分隔
    return "\n\n".join(md_parts).strip() + "\n"


# ------------------------------
# 可选：拼接调试 PDF（叠加可视化框）
# ------------------------------

def render_debug_pdf(pages_img: List[Image.Image], pages_blocks: List[List[ContentBlock]]) -> bytes:
    vis_pages = []
    for img, blocks in zip(pages_img, pages_blocks):
        vis = img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis)
        W, H = vis.size
        for b in blocks:
            x0, y0, x1, y1 = b.bbox
            box = (x0 * W, y0 * H, x1 * W, y1 * H)
            color = (0, 128, 255)
            draw.rectangle(box, outline=color, width=3)
        vis_pages.append(vis)
    # PIL 合并导出 PDF（内存）
    buf = io.BytesIO()
    if vis_pages:
        first, rest = vis_pages[0], vis_pages[1:]
        first.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()


# ------------------------------
# 主流程类
# ------------------------------

class VLMHttpToMarkdown:
    def __init__(self, server_url: str = VLM_SERVER_URL):
        self.client = VLMHttpClient(server_url)

    def _extract_page(self, img: Image.Image) -> List[ContentBlock]:
        blocks = self.client.infer_one_image(img)
        # 页内阅读顺序整理
        blocks = sort_blocks_reading_order(blocks)
        return blocks

    def from_images(self, images: List[Image.Image], *, make_debug_pdf: bool = False) -> Dict[str, Any]:
        pages_img = images
        pages_blocks: List[List[ContentBlock]] = [self._extract_page(im) for im in pages_img]
        # 跨页表格合并
        merge_cross_page_tables(pages_blocks)
        # 生成 Markdown
        md_pages = [page_to_markdown(img, blks) for img, blks in zip(pages_img, pages_blocks)]
        md = "\n".join(md_pages)
        # middle_json-like
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

    def from_pdf_bytes(self, pdf_bytes: bytes, *, dpi: int = 220, make_debug_pdf: bool = False) -> Dict[str, Any]:
        images = pdf_bytes_to_images(pdf_bytes, dpi=dpi)
        return self.from_images(images, make_debug_pdf=make_debug_pdf)


# ------------------------------
# 便捷函数
# ------------------------------

def build_markdown_from_pdf_bytes(pdf_bytes: bytes, *, dpi: int = 220, make_debug_pdf: bool = False) -> Dict[str, Any]:
    """最常用入口：给 PDF（二进制），返回 {markdown, middle_json[, debug_pdf_bytes]}"""
    return VLMHttpToMarkdown().from_pdf_bytes(pdf_bytes, dpi=dpi, make_debug_pdf=make_debug_pdf)


def build_markdown_from_images(images: List[Image.Image], *, make_debug_pdf: bool = False) -> Dict[str, Any]:
    """给多张图片：返回 {markdown, middle_json[, debug_pdf_bytes]}"""
    return VLMHttpToMarkdown().from_images(images, make_debug_pdf=make_debug_pdf)


if __name__ == "__main__":
    # 极简 CLI（仅示例，仍然保持“全内存”理念，不落盘）
    import argparse, sys, base64

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="输入文件路径：PDF 或 图片（png/jpg）")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--debug-pdf", action="store_true", help="返回叠加可视化框的调试 PDF（base64 打印）")
    args = parser.parse_args()

    path = args.input
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        out = build_markdown_from_pdf_bytes(pdf_bytes, dpi=args.dpi, make_debug_pdf=args.debug_pdf)
    else:
        img = Image.open(path)
        out = build_markdown_from_images([img], make_debug_pdf=args.debug_pdf)

    sys.stdout.write(out["markdown"])  # 直接打印 MD

    if args.debug_pdf and "debug_pdf_bytes" in out:
        b64 = base64.b64encode(out["debug_pdf_bytes"]).decode()
        sys.stderr.write("\n[debug_pdf_bytes_base64]" + b64 + "\n")
