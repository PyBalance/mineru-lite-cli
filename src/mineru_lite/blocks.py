from __future__ import annotations

import numpy as np
from typing import List

from .model import ContentBlock


def sort_blocks_reading_order(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """按列感知→从上到下、从左到右排序。"""
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
    ordered: List[ContentBlock] = []
    for _, col in col_with_key:
        ordered.extend(col)

    return ordered