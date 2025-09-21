from __future__ import annotations

from bs4 import BeautifulSoup
from typing import List, Tuple
import numpy as np

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


def _parse_table(html: str) -> tuple[list[list[str]], list[list[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    thead_rows: list[list[str]] = []
    tbody_rows: list[list[str]] = []

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


def _rebuild_table_html(thead_rows: list[list[str]], tbody_rows: list[list[str]]) -> str:
    def tr_html(cells: list[str], is_header: bool) -> str:
        tag = "th" if is_header else "td"
        # 避免嵌套 HTML：仅取纯文本
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


def merge_cross_page_tables(pages: list[list]) -> None:
    """就地合并跨页表格：最后一张表 + 下一页第一张表。"""
    from .model import ContentBlock  # 延迟导入以避免循环

    for p in range(1, len(pages)):
        prev_blocks = [b for b in pages[p - 1] if isinstance(b, ContentBlock) and b.type == "table"]
        curr_blocks = [b for b in pages[p] if isinstance(b, ContentBlock) and b.type == "table"]
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

        def rows_text(rows: list[list[str]]) -> list[str]:
            return ["|".join(r) for r in rows]

        new_thead = (
            p_thead if (p_thead and c_thead and rows_text(p_thead) == rows_text(c_thead)) else (p_thead or c_thead)
        )
        new_tbody = p_tbody + c_tbody
        prev.content = _rebuild_table_html(new_thead, new_tbody)
        curr.type = "__deleted_table__"