from __future__ import annotations

import io
import os
import sys
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
from bs4 import BeautifulSoup
from platformdirs import PlatformDirs

# 兼容 tomllib / tomli、tomli_w
try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
import tomli_w  # 写 TOML

try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None

try:
    from mineru_vl_utils import MinerUClient
except Exception as e:
    raise RuntimeError(
        "请先安装 mineru-vl-utils： pip install mineru-vl-utils==0.1.8\n"
        f"ImportError: {e}"
    )

# ------------------------------
# 默认常量
# ------------------------------
DEFAULT_SERVER_URL = "http://192.168.191.126:30000"
APP_NAME = "mineru-lite"  # 用于配置目录名与命令名呼应

# ------------------------------
# 页码解析工具函数
# ------------------------------
def parse_pages_spec(spec: Optional[str], total_pages: int) -> List[int]:
    """
    spec: 例如 '1,2', '1-3', '3-', '-3', '1,2,5-6', '2,3,6-'
    返回：升序且去重的 0-based 页索引列表。
    """
    if not spec or not spec.strip():
        return list(range(total_pages))  # 全部
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
        # 1-based → clamp → 0-based
        l = max(1, min(l, total_pages))
        r = max(1, min(r, total_pages))
        if l <= r:
            for p in range(l, r + 1):
                result.add(p - 1)
        else:
            # 不支持倒序，忽略；如需支持可在此处理
            continue
    return sorted(result)

# ------------------------------
# 缓存基础设施
# ------------------------------
import hashlib, json, tempfile

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def cache_dir() -> Path:
    d = PlatformDirs(appname=APP_NAME, appauthor=False)
    p = Path(d.user_cache_path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def cache_path_for_hash(h: str) -> Path:
    return cache_dir() / f"{h}.json"

def load_pdf_cache(file_hash: str) -> dict:
    fp = cache_path_for_hash(file_hash)
    if not fp.exists(): return {}
    try:
        return json.loads(fp.read_text("utf-8"))
    except Exception:
        return {}

def save_pdf_cache_atomic(file_hash: str, obj: dict) -> None:
    fp = cache_path_for_hash(file_hash)
    tmp = Path(tempfile.mkstemp(prefix="cache-", suffix=".json", dir=str(fp.parent))[1])
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        tmp.replace(fp)  # 原子替换
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except Exception: pass

# ------------------------------
# 配置加载 / 保存
# ------------------------------
def _config_path() -> Path:
    d = PlatformDirs(appname=APP_NAME, appauthor=False)
    p = Path(d.user_config_path)
    p.mkdir(parents=True, exist_ok=True)
    return p / "config.toml"

def load_config() -> Dict[str, Any]:
    fp = _config_path()
    if not fp.exists():
        return {}
    try:
        with fp.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}

def save_config(cfg: Dict[str, Any]) -> None:
    fp = _config_path()
    with fp.open("wb") as f:
        tomli_w.dump(cfg, f)

def interactive_config(existing: Dict[str, Any]) -> None:
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
    env_server = os.getenv("MINERU_LITE_SERVER_URL")
    cfg_server = load_config().get("server_url")
    return cli_server or env_server or cfg_server or DEFAULT_SERVER_URL

# ------------------------------
# 你的原始实现（略：核心类与函数保持不变，只把默认 URL 换成参数传递）
# ------------------------------
VLM_SERVER_URL = DEFAULT_SERVER_URL  # 仍然保留一个默认值

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
        c = getattr(x, "content", None) if hasattr(x, "content") else x.get("content")
        return ContentBlock(type=t, bbox=tuple(b), angle=a, content=c)
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)
    def mid(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 220, page_indices: Optional[List[int]] = None) -> List[Image.Image]:
    if pdfium is None:
        raise RuntimeError("未安装 pypdfium2，无法渲 PDF；请 pip install pypdfium2")
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
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.client = MinerUClient(backend="http-client", server_url=server_url)
    def infer_one_image(self, image: Image.Image) -> List[ContentBlock]:
        blocks = self.client.two_step_extract(image)
        return [ContentBlock.from_any(b) for b in blocks]

def sort_blocks_reading_order(blocks: List[ContentBlock]) -> List[ContentBlock]:
    if not blocks:
        return blocks
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
    col_with_key = []
    for col in columns:
        x0_mean = float(np.mean([b.bbox[0] for b in col])) if col else 0.0
        col_with_key.append((x0_mean, col))
    col_with_key.sort(key=lambda t: t[0])
    ordered = []
    for _, col in col_with_key:
        ordered.extend(col)
    return ordered

_HTML_UNESCAPE_MAP = {"&amp;lt;": "<", "&amp;gt;": ">", "&amp;le;": "≤", "&amp;ge;": "≥"}
def html_table_filter(html: str) -> str:
    for k, v in _HTML_UNESCAPE_MAP.items():
        html = html.replace(k, v)
    return html

def _parse_table(html: str):
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
        for tr in soup.find_all("tr"):
            tbody_rows.append([c.get_text(strip=True) for c in tr.find_all(["th", "td"])])
    return thead_rows, tbody_rows

def _count_columns(rows: List[List[str]]) -> int:
    return int(np.median([len(r) for r in rows])) if rows else 0

def _rebuild_table_html(thead_rows, tbody_rows) -> str:
    def tr_html(cells, is_header):
        tag = "th" if is_header else "td"
        from bs4 import BeautifulSoup as BS
        return "<tr>" + "".join(f"<{tag}>{BS(c, 'html.parser').get_text()}</{tag}>" for c in cells) + "</tr>"
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
        def rows_text(rows): return ["|".join(r) for r in rows]
        new_thead = p_thead if (p_thead and c_thead and rows_text(p_thead) == rows_text(c_thead)) else (p_thead or c_thead)
        new_tbody = p_tbody + c_tbody
        prev.content = _rebuild_table_html(new_thead, new_tbody)
        curr.type = "__deleted_table__"

def page_to_markdown(page_img: Image.Image, blocks: List[ContentBlock]) -> str:
    md_parts: List[str] = []
    for b in blocks:
        if b.type == "__deleted_table__": continue
        if b.type == "text":
            text = (b.content or "").strip()
            if text: md_parts.append(text)
        elif b.type == "equation":
            tex = (b.content or "").strip()
            if tex: md_parts.append(f"$\n{tex}\n$")
        elif b.type == "table":
            html = html_table_filter(b.content or "")
            if html: md_parts.append("\n" + html + "\n")
        elif b.type == "image":
            md_parts.append("（图片略）")
        else:
            if b.content: md_parts.append(b.content)
    return "\n\n".join(md_parts).strip() + "\n"

def render_debug_pdf(pages_img: List[Image.Image], pages_blocks: List[List[ContentBlock]]) -> bytes:
    vis_pages = []
    for img, blocks in zip(pages_img, pages_blocks):
        vis = img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis)
        W, H = vis.size
        for b in blocks:
            x0, y0, x1, y1 = b.bbox
            box = (x0 * W, y0 * H, x1 * W, y1 * H)
            draw.rectangle(box, outline=(0,128,255), width=3)
        vis_pages.append(vis)
    buf = io.BytesIO()
    if vis_pages:
        first, rest = vis_pages[0], vis_pages[1:]
        first.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()

class VLMHttpToMarkdown:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
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
                    {"width": img.size[0], "height": img.size[1], "blocks": [b.__dict__ for b in blks]}
                    for img, blks in zip(pages_img, pages_blocks)
                ],
            }
        }
        result = {"markdown": md, "middle_json": middle}
        if make_debug_pdf:
            result["debug_pdf_bytes"] = render_debug_pdf(pages_img, pages_blocks)
        return result
    def from_pdf_bytes(self, pdf_bytes: bytes, *, dpi: int = 220, make_debug_pdf: bool = False,
                   pages_spec: Optional[str] = None) -> Dict[str, Any]:
    # 1) 基本信息
    file_hash = sha256_bytes(pdf_bytes)
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    total_pages = len(pdf)
    wanted = parse_pages_spec(pages_spec, total_pages)  # 0-based 升序

    # 2) 读取/准备缓存
    cache = load_pdf_cache(file_hash)
    if not cache or cache.get("dpi") != dpi:
        cache = {"version": 1, "file_sha256": file_hash, "page_count": total_pages, "dpi": dpi, "pages": {}}
    pages_cache: dict = cache.setdefault("pages", {})

    # 3) 已缓存 & 待补页
    cached_have = sorted(int(k) for k in pages_cache.keys())
    missing = [i for i in wanted if str(i) not in pages_cache]

    # 4) 对缺页：渲图 + 推理 + 存缓存（只渲缺页，减少 IO）
    if missing:
        # 渲图（缺页）
        scale = dpi / 72.0
        for i in missing:
            page = pdf[i]
            try:
                pil = page.render(scale=scale).to_pil()
            except AttributeError:
                pil = page.render_to(pdfium.BitmapConv.pil_image, scale=scale)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")

            # 推理（排序留在 _extract_page 里）
            blocks = self._extract_page(pil)
            pages_cache[str(i)] = {
                "width": pil.size[0],
                "height": pil.size[1],
                "blocks": [b.__dict__ for b in blocks],
            }

        save_pdf_cache_atomic(file_hash, cache)

    # 5) 组装本次所需的图与块（从缓存构造 ContentBlock，不再触发后端）
    pages_img: List[Image.Image] = []
    pages_blocks: List[List[ContentBlock]] = []
    scale = dpi / 72.0
    for i in wanted:
        meta = pages_cache[str(i)]
        # 如果需要 debug PDF/或保持接口一致，渲图本页；否则也可以按需渲
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

    # 6) 跨页表格合并（在当前实例的副本上，不反写缓存）
    merge_cross_page_tables(pages_blocks)

    # 7) 生成 Markdown
    md_pages = [page_to_markdown(img, blks) for img, blks in zip(pages_img, pages_blocks)]
    md = "\n".join(md_pages)
    middle = {
        "pdf_info": {
            "page_count": len(pages_img),
            "pages": [
                {"width": img.size[0], "height": img.size[1], "blocks": [b.__dict__ for b in blks]}
                for img, blks in zip(pages_img, pages_blocks)
            ],
        }
    }
    result: Dict[str, Any] = {"markdown": md, "middle_json": middle}
    if make_debug_pdf:
        result["debug_pdf_bytes"] = render_debug_pdf(pages_img, pages_blocks)
    return result

def build_markdown_from_pdf_bytes(pdf_bytes: bytes, *, dpi: int = 220,
                                    make_debug_pdf: bool = False, server_url: str = DEFAULT_SERVER_URL,
                                    pages_spec: Optional[str] = None):
    return VLMHttpToMarkdown(server_url=server_url).from_pdf_bytes(
        pdf_bytes, dpi=dpi, make_debug_pdf=make_debug_pdf, pages_spec=pages_spec
    )

def build_markdown_from_images(images: List[Image.Image], *, make_debug_pdf: bool = False, server_url: str = DEFAULT_SERVER_URL):
    return VLMHttpToMarkdown(server_url=server_url).from_images(images, make_debug_pdf=make_debug_pdf)

# ------------------------------
# CLI：子命令 + 交互配置
# ------------------------------
def _add_run_args(subparser):
    subparser.add_argument("input", help="输入文件路径：PDF 或 图片（png/jpg）")
    subparser.add_argument("--dpi", type=int, default=220)
    subparser.add_argument("--debug-pdf", action="store_true", help="返回叠加可视化框的调试 PDF（base64 打印）")
    subparser.add_argument("--server", help="覆盖服务端 URL（优先级最高）")
    subparser.add_argument("--configure", action="store_true", help="运行前进入交互式配置向导")
    subparser.add_argument("--pages", help="页码选择（1-based）：例如 1,2 或 1-3 或 3- 或 -3 或 1,2,5-6", default=None)

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="mineru-lite", description="VLM http-client → Markdown（轻量 CLI）")
    subparsers = parser.add_subparsers(dest="command")

    # 默认 run（无子命令时也走它）
    run_p = subparsers.add_parser("run", help="运行转换")
    _add_run_args(run_p)

    # config 子命令
    cfg_p = subparsers.add_parser("config", help="查看/设置/重置 配置")
    cfg_sub = cfg_p.add_subparsers(dest="cfg_cmd")
    cfg_sub.add_parser("show", help="显示当前配置")
    set_p = cfg_sub.add_parser("set", help="设置配置项")
    set_p.add_argument("--server", required=False, help="设置 server_url")
    cfg_sub.add_parser("reset", help="重置（删除配置文件）")
    cfg_sub.add_parser("wizard", help="交互式配置向导（同 --configure）")

    # 兼容：直接 mineru-lite <args> 也认为是 run
    args, extras = parser.parse_known_args(argv)
    if args.command is None:
        # 注入成 run 子命令解析
        args = parser.parse_args(["run", *(argv or [])])

    # 处理 config 子命令
    if args.command == "config":
        cmd = args.cfg_cmd
        if cmd == "show":
            print(f"配置路径：{_config_path()}")
            print(load_config() or "{}")
            return
        if cmd == "set":
            cfg = load_config()
            if getattr(args, "server", None):
                cfg["server_url"] = args.server
                save_config(cfg)
                print("已写入 server_url。")
            else:
                print("未提供 --server。")
            return
        if cmd == "reset":
            fp = _config_path()
            if fp.exists(): fp.unlink()
            print("已删除配置文件。")
            return
        if cmd == "wizard" or cmd is None:
            interactive_config(load_config())
            return

    # run 流程
    if getattr(args, "configure", False):
        interactive_config(load_config())

    server_url = resolve_server(getattr(args, "server", None))
    path = args.input

    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        out = build_markdown_from_pdf_bytes(
            pdf_bytes, dpi=args.dpi, make_debug_pdf=args.debug_pdf,
            server_url=server_url, pages_spec=args.pages
        )
    else:
        img = Image.open(path)
        out = build_markdown_from_images([img], make_debug_pdf=args.debug_pdf, server_url=server_url)

    sys.stdout.write(out["markdown"])
    if args.debug_pdf and "debug_pdf_bytes" in out:
        b64 = base64.b64encode(out["debug_pdf_bytes"]).decode()
        sys.stderr.write("\n[debug_pdf_bytes_base64]" + b64 + "\n")

if __name__ == "__main__":
    main()