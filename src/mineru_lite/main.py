"""
MinerU Lite - VLM HTTP Client for PDF/Image to Markdown Conversion

A lightweight command-line tool that uses a VLM (Vision Language Model) HTTP service
to convert PDF documents and images to structured markdown format.
"""

from __future__ import annotations

import io
import os
import sys
import base64
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
import numpy as np
from bs4 import BeautifulSoup
from platformdirs import PlatformDirs

# Import configuration libraries with fallback for different Python versions
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
import tomli_w

# Import PDF library with fallback
try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None

# Import VLM client
try:
    from mineru_vl_utils import MinerUClient
except Exception as e:
    raise RuntimeError(
        "Please install mineru-vl-utils: pip install mineru-vl-utils==0.1.8\n"
        f"ImportError: {e}"
    )

# Configuration constants
DEFAULT_SERVER_URL = "http://127.0.0.1:30000"
APP_NAME = "mineru-lite"


def parse_pages_spec(spec: Optional[str], total_pages: int) -> List[int]:
    """
    Parse page specification string into list of 0-based page indices.

    Args:
        spec: Page specification like '1,2', '1-3', '3-', '-3', '1,2,5-6', '2,3,6-'
        total_pages: Total number of pages in the document

    Returns:
        Sorted list of unique 0-based page indices
    """
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

        # Convert 1-based to 0-based with clamping
        l = max(1, min(l, total_pages))
        r = max(1, min(r, total_pages))

        if l <= r:
            for p in range(l, r + 1):
                result.add(p - 1)

    return sorted(result)


def sha256_bytes(data: bytes) -> str:
    """Calculate SHA256 hash of bytes data."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def cache_dir() -> Path:
    """Get the cache directory path."""
    d = PlatformDirs(appname=APP_NAME, appauthor=False)
    p = Path(d.user_cache_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_path_for_hash(file_hash: str) -> Path:
    """Get cache file path for a given file hash."""
    return cache_dir() / f"{file_hash}.json"


def load_pdf_cache(file_hash: str) -> dict:
    """Load cached PDF processing data."""
    fp = cache_path_for_hash(file_hash)
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text("utf-8"))
    except Exception:
        return {}


def save_pdf_cache_atomic(file_hash: str, obj: dict) -> None:
    """Save PDF cache data atomically."""
    fp = cache_path_for_hash(file_hash)
    tmp = Path(tempfile.mkstemp(prefix="cache-", suffix=".json", dir=str(fp.parent))[1])
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        tmp.replace(fp)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _config_path() -> Path:
    """Get configuration file path."""
    d = PlatformDirs(appname=APP_NAME, appauthor=False)
    p = Path(d.user_config_path)
    p.mkdir(parents=True, exist_ok=True)
    return p / "config.toml"


def load_config() -> Dict[str, Any]:
    """Load configuration from TOML file."""
    fp = _config_path()
    if not fp.exists():
        return {}
    try:
        with fp.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    """Save configuration to TOML file."""
    fp = _config_path()
    with fp.open("wb") as f:
        tomli_w.dump(cfg, f)


def interactive_config(existing: Dict[str, Any]) -> None:
    """Run interactive configuration wizard."""
    print(f"[配置向导] 当前配置文件：{_config_path()}")
    cur_server = existing.get("server_url", "") or ""
    print(f"当前 server_url：{cur_server or '(未设置，使用默认)'}")
    new_server = input("请输入新的 server_url（回车跳过保留现状）：").strip()

    if new_server:
        cfg = dict(existing)
        cfg["server_url"] = new_server
        save_config(cfg)
        print("已保存到配置文件。")
    else:
        print("未更改。")


def resolve_server(cli_server: Optional[str]) -> str:
    """Resolve server URL from CLI argument, environment variable, or config file."""
    import re

    env_server = os.getenv("MINERU_LITE_SERVER_URL")
    cfg_server = load_config().get("server_url")
    server_url = cli_server or env_server or cfg_server or DEFAULT_SERVER_URL

    # Automatically add http:// prefix if URL lacks protocol scheme
    if server_url and not re.match(r'^https?://', server_url):
        server_url = f"http://{server_url}"

    return server_url


@dataclass
class ContentBlock:
    """Represents a content block extracted from a document."""
    type: str
    bbox: Tuple[float, float, float, float]
    angle: Optional[int]
    content: Optional[str]

    @staticmethod
    def from_any(x: Any) -> "ContentBlock":
        """Create ContentBlock from any object with compatible attributes."""
        t = getattr(x, "type", None) or x.get("type")
        b = getattr(x, "bbox", None) or x.get("bbox")
        a = getattr(x, "angle", None) if hasattr(x, "angle") else x.get("angle")
        c = getattr(x, "content", None) if hasattr(x, "content") else x.get("content")
        return ContentBlock(type=t, bbox=tuple(b), angle=a, content=c)

    def area(self) -> float:
        """Calculate area of the bounding box."""
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def mid(self) -> Tuple[float, float]:
        """Get midpoint of the bounding box."""
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def pdf_bytes_to_images(
    pdf_bytes: bytes,
    dpi: int = 220,
    page_indices: Optional[List[int]] = None
) -> List[Image.Image]:
    """Convert PDF bytes to list of PIL Images."""
    if pdfium is None:
        raise RuntimeError("pypdfium2 not installed, cannot render PDF. Please: pip install pypdfium2")

    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    total = len(pdf)

    if page_indices is None:
        indices = range(total)
    else:
        indices = page_indices

    images: List[Image.Image] = []
    scale = dpi / 72.0

    for i in indices:
        page = pdf[i]
        try:
            pil = page.render(scale=scale).to_pil()
        except AttributeError:
            pil = page.render_to(pdfium.BitmapConv.pil_image, scale=scale)

        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        images.append(pil)

    return images


class VLMHttpClient:
    """HTTP client for VLM service."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.client = MinerUClient(backend="http-client", server_url=server_url)

    def infer_one_image(self, image: Image.Image) -> List[ContentBlock]:
        """Extract content blocks from a single image."""
        blocks = self.client.two_step_extract(image)
        return [ContentBlock.from_any(b) for b in blocks]


def sort_blocks_reading_order(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """Sort blocks into reading order (column-aware)."""
    if not blocks:
        return blocks

    # Group blocks by columns based on horizontal position
    mids = np.array([b.mid()[0] for b in blocks])
    mids_sorted_idx = np.argsort(mids)
    mids_sorted = mids[mids_sorted_idx]
    gaps = np.diff(mids_sorted)
    cut_points = np.where(gaps > 0.15)[0]

    if len(cut_points) == 0:
        columns = [blocks]
    else:
        parts = np.split(mids_sorted_idx, cut_points + 1)
        columns = [[blocks[i] for i in part] for part in parts]

    def sort_in_column(col: List[ContentBlock]) -> List[ContentBlock]:
        return sorted(col, key=lambda b: (b.bbox[1], b.bbox[0]))

    columns = [sort_in_column(col) for col in columns]

    # Sort columns by average x position
    col_with_key = []
    for col in columns:
        x0_mean = float(np.mean([b.bbox[0] for b in col])) if col else 0.0
        col_with_key.append((x0_mean, col))

    col_with_key.sort(key=lambda t: t[0])
    ordered = []
    for _, col in col_with_key:
        ordered.extend(col)

    return ordered


_HTML_UNESCAPE_MAP = {
    "&amp;lt;": "<",
    "&amp;gt;": ">",
    "&amp;le;": "≤",
    "&amp;ge;": "≥",
}


def html_table_filter(html: str) -> str:
    """Filter and clean HTML table content."""
    for k, v in _HTML_UNESCAPE_MAP.items():
        html = html.replace(k, v)
    return html


def _parse_table(html: str):
    """Parse HTML table into header and body rows."""
    soup = BeautifulSoup(html, "html.parser")
    thead_rows, tbody_rows = [], []

    thead = soup.find("thead")
    if thead:
        for tr in thead.find_all("tr"):
            thead_rows.append(
                [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            )

    tbody = soup.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            tbody_rows.append(
                [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            )
    else:
        for tr in soup.find_all("tr"):
            tbody_rows.append(
                [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            )

    return thead_rows, tbody_rows


def _count_columns(rows: List[List[str]]) -> int:
    """Count median number of columns in rows."""
    return int(np.median([len(r) for r in rows])) if rows else 0


def _rebuild_table_html(thead_rows, tbody_rows) -> str:
    """Rebuild table HTML from parsed rows."""
    def tr_html(cells, is_header):
        tag = "th" if is_header else "td"
        from bs4 import BeautifulSoup as BS

        return (
            "<tr>"
            + "".join(
                f"<{tag}>{BS(c, 'html.parser').get_text()}</{tag}>" for c in cells
            )
            + "</tr>"
        )

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
    """Merge tables that span across page boundaries."""
    for p in range(1, len(pages)):
        prev_blocks = [b for b in pages[p - 1] if b.type == "table"]
        curr_blocks = [b for b in pages[p] if b.type == "table"]

        if not prev_blocks or not curr_blocks:
            continue

        prev = prev_blocks[-1]
        curr = curr_blocks[0]

        if not prev.content or not curr.content:
            continue

        p_thead, p_tbody = _parse_table(html_table_filter(prev.content))
        c_thead, c_tbody = _parse_table(html_table_filter(curr.content))
        p_cols = _count_columns(p_thead or p_tbody)
        c_cols = _count_columns(c_thead or c_tbody)

        if p_cols == 0 or c_cols == 0 or p_cols != c_cols:
            continue

        def rows_text(rows):
            return ["|".join(r) for r in rows]

        new_thead = (
            p_thead
            if (p_thead and c_thead and rows_text(p_thead) == rows_text(c_thead))
            else (p_thead or c_thead)
        )
        new_tbody = p_tbody + c_tbody
        prev.content = _rebuild_table_html(new_thead, new_tbody)
        curr.type = "__deleted_table__"


def page_to_markdown(page_img: Image.Image, blocks: List[ContentBlock]) -> str:
    """Convert content blocks to markdown format."""
    md_parts: List[str] = []

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


def render_debug_pdf(
    pages_img: List[Image.Image],
    pages_blocks: List[List[ContentBlock]]
) -> bytes:
    """Render debug PDF with bounding boxes overlaid."""
    vis_pages = []

    for img, blocks in zip(pages_img, pages_blocks):
        vis = img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis)
        W, H = vis.size

        for b in blocks:
            x0, y0, x1, y1 = b.bbox
            box = (x0 * W, y0 * H, x1 * W, y1 * H)
            draw.rectangle(box, outline=(0, 128, 255), width=3)

        vis_pages.append(vis)

    buf = io.BytesIO()
    if vis_pages:
        first, rest = vis_pages[0], vis_pages[1:]
        first.save(buf, format="PDF", save_all=True, append_images=rest)

    return buf.getvalue()


class VLMHttpToMarkdown:
    """Main class for converting documents to markdown using VLM service."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.client = VLMHttpClient(server_url)

    def _extract_page(self, img: Image.Image) -> List[ContentBlock]:
        """Extract and sort content blocks from a page image."""
        blocks = self.client.infer_one_image(img)
        return sort_blocks_reading_order(blocks)

    def from_images(
        self,
        images: List[Image.Image],
        *,
        make_debug_pdf: bool = False
    ) -> Dict[str, Any]:
        """Convert list of images to markdown."""
        pages_img = images
        pages_blocks = [self._extract_page(im) for im in pages_img]
        merge_cross_page_tables(pages_blocks)

        md_pages = [
            page_to_markdown(img, blks) for img, blks in zip(pages_img, pages_blocks)
        ]
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

        result = {"markdown": md, "middle_json": middle}

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
        """Convert PDF bytes to markdown with caching."""
        # Basic information
        file_hash = sha256_bytes(pdf_bytes)
        pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
        total_pages = len(pdf)
        wanted = parse_pages_spec(pages_spec, total_pages)

        # Load/prepare cache
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

        # Find cached and missing pages
        cached_have = sorted(int(k) for k in pages_cache.keys())
        missing = [i for i in wanted if str(i) not in pages_cache]

        # Process missing pages: render + inference + cache
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

        # Assemble images and blocks for this run
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

        # Merge cross-page tables (on current instance copy, don't write back to cache)
        merge_cross_page_tables(pages_blocks)

        # Generate markdown
        md_pages = [
            page_to_markdown(img, blks) for img, blks in zip(pages_img, pages_blocks)
        ]
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
    server_url: str = DEFAULT_SERVER_URL,
    pages_spec: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to convert PDF bytes to markdown."""
    return VLMHttpToMarkdown(server_url=server_url).from_pdf_bytes(
        pdf_bytes, dpi=dpi, make_debug_pdf=make_debug_pdf, pages_spec=pages_spec
    )


def build_markdown_from_images(
    images: List[Image.Image],
    *,
    make_debug_pdf: bool = False,
    server_url: str = DEFAULT_SERVER_URL,
) -> Dict[str, Any]:
    """Convenience function to convert images to markdown."""
    return VLMHttpToMarkdown(server_url=server_url).from_images(
        images, make_debug_pdf=make_debug_pdf
    )


def _add_run_args(subparser):
    """Add run command arguments to subparser."""
    subparser.add_argument("input", help="输入文件路径：PDF 或 图片（png/jpg）")
    subparser.add_argument("--dpi", type=int, default=220)
    subparser.add_argument(
        "--debug-pdf",
        action="store_true",
        help="返回叠加可视化框的调试 PDF（base64 打印）",
    )
    subparser.add_argument("--server", help="覆盖服务端 URL（优先级最高）")
    subparser.add_argument(
        "--configure", action="store_true", help="运行前进入交互式配置向导"
    )
    subparser.add_argument(
        "--pages",
        help="页码选择（1-based）：例如 1,2 或 1-3 或 3- 或 -3 或 1,2,5-6",
        default=None,
    )

def _add_summary_args(subparser):
    """Add summary command arguments to subparser."""
    subparser.add_argument("input", help="输入文件路径：仅支持 PDF")
    subparser.add_argument(
        "--pages",
        help="页码选择（1-based）：例如 1,2 或 1-3 或 3- 或 -3 或 1,2,5-6；缺省为全部页",
        default=None,
    )

def _human_bytes(n: int) -> str:
    """Format bytes into human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    i = 0
    while s >= 1024 and i < len(units) - 1:
        s /= 1024.0
        i += 1
    # 保留两位（去掉多余的 0）
    out = f"{s:.2f}".rstrip("0").rstrip(".")
    return f"{out} {units[i]}"

def _normalize_text(t: str) -> str:
    """Normalize whitespace for preview."""
    import re
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # 压缩多空格
    t = re.sub(r"[ \t]+", " ", t)
    # 限制连续空行
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def summarize_pdf(path: str, pages_spec: Optional[str]) -> None:
    """Print a PDF summary using only its text layer (no backend)."""
    if pdfium is None:
        raise RuntimeError("需要 pypdfium2：pip install pypdfium2")
    if not path.lower().endswith(".pdf"):
        raise RuntimeError("summary 仅支持 PDF 文件。")

    # 文件整体信息
    file_size = os.path.getsize(path)
    pdf = pdfium.PdfDocument(path)
    total_pages = len(pdf)

    wanted = parse_pages_spec(pages_spec, total_pages)
    # 统计总字符数（含空白）
    total_chars = 0

    # 头部
    print(f"{os.path.basename(path)} 的 summary")
    print(f"文件大小: {_human_bytes(file_size)}")
    print(f"总页数: {total_pages}")

    # 先遍历统计每页，顺便累计总字符
    per_page_info = []
    for i in wanted:
        page = pdf[i]
        width_pt, height_pt = page.get_size()  # pt
        # mm 近似
        mm_per_pt = 25.4 / 72.0
        w_mm = width_pt * mm_per_pt
        h_mm = height_pt * mm_per_pt
        orient = "纵向" if height_pt >= width_pt else "横向"

        text = ""
        page_chars = 0
        try:
            tp = page.get_textpage()
            try:
                # 方式1：按总字符数读取（优先）
                n = tp.count_chars()
                page_chars = int(n)
                text = tp.get_text_range(0, n) if n > 0 else ""
            except Exception:
                # 方式2：回退到全页包围盒
                try:
                    text = tp.get_text_bounded(0, 0, width_pt, height_pt)  # 若 API 支持
                    page_chars = len(text)
                except Exception:
                    text = ""
                    page_chars = 0
        except Exception:
            # 无法提取文字（如受保护/扫描件）
            text = ""
            page_chars = 0

        total_chars += page_chars
        per_page_info.append({
            "index": i + 1,  # 1-based
            "w_pt": width_pt, "h_pt": height_pt,
            "w_mm": w_mm, "h_mm": h_mm,
            "orient": orient,
            "chars": page_chars,
            "text": text,
        })

    print(f"总字符数（文字层，含空白）: {total_chars}")
    print()

    # 逐页输出
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
        # 为了阅读清晰，单独标头
        print("文字预览：")
        print("开头[最多100字]:")
        print(head)
        print()
        print("结尾[最多30字]:")
        print(tail)
        print()


def main(argv=None):
    """Main entry point for the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mineru-lite",
        description="VLM http-client → Markdown（轻量 CLI）"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default run command
    run_p = subparsers.add_parser("run", help="运行转换")
    _add_run_args(run_p)

    # Summary command
    sum_p = subparsers.add_parser("summary", help="仅基于 PDF 文字层的摘要（不调用后端）")
    _add_summary_args(sum_p)

    # Config command
    cfg_p = subparsers.add_parser("config", help="查看/设置/重置 配置")
    cfg_sub = cfg_p.add_subparsers(dest="cfg_cmd")
    cfg_sub.add_parser("show", help="显示当前配置")
    set_p = cfg_sub.add_parser("set", help="设置配置项")
    set_p.add_argument("--server", required=False, help="设置 server_url")
    cfg_sub.add_parser("reset", help="重置（删除配置文件）")
    cfg_sub.add_parser("wizard", help="交互式配置向导（同 --configure）")

    # Compatibility: direct args without subcommand
    args, extras = parser.parse_known_args(argv)
    if args.command is None:
        args = parser.parse_args(["run", *(argv or [])])

    # Handle config commands
    if args.command == "config":
        cmd = args.cfg_cmd
        if cmd == "show":
            print(f"配置路径：{_config_path()}")
            print(load_config() or "{}")
            return
        elif cmd == "set":
            cfg = load_config()
            if getattr(args, "server", None):
                cfg["server_url"] = args.server
                save_config(cfg)
                print("已写入 server_url。")
            else:
                print("未提供 --server。")
            return
        elif cmd == "reset":
            fp = _config_path()
            if fp.exists():
                fp.unlink()
            print("已删除配置文件。")
            return
        elif cmd == "wizard" or cmd is None:
            interactive_config(load_config())
            return

    # Summary workflow
    if args.command == "summary":
        path = args.input
        summarize_pdf(path, getattr(args, "pages", None))
        return

    # Run workflow
    if getattr(args, "configure", False):
        interactive_config(load_config())

    server_url = resolve_server(getattr(args, "server", None))
    path = args.input

    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        out = build_markdown_from_pdf_bytes(
            pdf_bytes,
            dpi=args.dpi,
            make_debug_pdf=args.debug_pdf,
            server_url=server_url,
            pages_spec=args.pages,
        )
    else:
        img = Image.open(path)
        out = build_markdown_from_images(
            [img], make_debug_pdf=args.debug_pdf, server_url=server_url
        )

    sys.stdout.write(out["markdown"])
    if args.debug_pdf and "debug_pdf_bytes" in out:
        b64 = base64.b64encode(out["debug_pdf_bytes"]).decode()
        sys.stderr.write("\n[debug_pdf_bytes_base64]" + b64 + "\n")


if __name__ == "__main__":
    main()
