"""Question segmentation — split the page into numbered question cells.

Primary signal: printed-text horizontal projection profile (row ink
density). Ruling lines, when present, are detected with HoughLinesP and
used as hard separators. Each row is split into (question_area,
answer_area) by a vertical projection valley or a printed equals/blank run.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_ROW_INK = 6          # ink pixels per row to count as content
MIN_ROW_HEIGHT = 18      # px — question text lines are ~24px tall
GAP_MIN = 14             # blank rows that constitute a separator


def detect_rulings(printed: np.ndarray) -> list[int]:
    """Return y-coordinates of strong horizontal ruling lines."""
    edges = cv2.Canny(printed, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                            minLineLength=printed.shape[1] // 2, maxLineGap=8)
    ys = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if abs(y2 - y1) <= 3:
                ys.append(int((y1 + y2) / 2))
    ys.sort()
    merged = []
    for y in ys:
        if not merged or y - merged[-1] > 10:
            merged.append(y)
    return merged


def projection_rows(printed: np.ndarray) -> list[tuple[int, int]]:
    """Content rows [y0, y1) from the horizontal projection of printed ink."""
    proj = np.count_nonzero(printed, axis=1)
    rows, y = [], 0
    h = len(proj)
    while y < h:
        if proj[y] >= MIN_ROW_INK:
            y0 = y
            gap = 0
            while y < h and gap <= GAP_MIN:
                if proj[y] >= MIN_ROW_INK:
                    gap = 0
                else:
                    gap += 1
                y += 1
            y1 = y - gap
            if y1 - y0 >= MIN_ROW_HEIGHT:
                rows.append((y0, y1))
        else:
            y += 1
    # drop rows that look like a single ruling line (very thin + extremely dense)
    return [(y0, y1) for y0, y1 in rows
            if not (y1 - y0 < 12 and np.count_nonzero(printed[y0:y1]) >
                    0.8 * printed.shape[1] * (y1 - y0))]


def split_columns(row_printed: np.ndarray):
    """Split a row crop into (question_part, answer_part) by the widest
    vertical whitespace valley in its right half."""
    w = row_printed.shape[1]
    proj = np.count_nonzero(row_printed, axis=0)
    # smooth
    k = 9
    sm = np.convolve(proj, np.ones(k) / k, mode="same")
    lo, hi = int(w * 0.35), int(w * 0.92)
    if hi - lo < 40:
        return row_printed, None
    valley = lo + int(np.argmin(sm[lo:hi]))
    if sm[valley] > 0.25 * sm[lo:hi].max():
        return row_printed, None  # no clear split; whole row is the answer zone
    return row_printed[:, :valley], row_printed[:, valley:]


def answer_zone(row_printed: np.ndarray, row_hw: np.ndarray, margin: int = 3):
    """The child writes after the printed '=' — cut the handwriting mask at
    the right edge of printed content. Ruling lines span the whole row and
    would push that edge to the paper border, so long horizontal runs are
    masked out first. Returns (q_part_printed, answer_hw)."""
    cols = np.count_nonzero(row_printed, axis=0)
    row_runs = np.count_nonzero(row_printed, axis=1)
    width = row_printed.shape[1]
    is_ruling = row_runs > 0.6 * width
    eff = row_printed.copy()
    eff[is_ruling, :] = 0
    cols_eff = np.count_nonzero(eff, axis=0)
    xs = np.nonzero(cols_eff)[0]
    if not xs.size:
        return row_printed, np.zeros_like(row_hw)
    x_end = int(xs.max())
    return eff, row_hw[:, x_end + margin:]


def _looks_like_question(row_printed: np.ndarray) -> bool:
    """Question rows start with a narrow blob (the printed 'N.' question
    number). Header/title rows start with a wide dense block (CJK title)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray((row_printed > 0).astype(np.uint8)), 8)
    boxes = [tuple(stats[i][:4]) for i in range(1, n) if stats[i][4] >= 8]
    if not boxes:
        return False
    x, y, w, h = min(boxes, key=lambda b: b[0])
    return w <= 38


def segment_questions(page_bgr: np.ndarray, printed: np.ndarray,
                      handwriting: np.ndarray, use_rulings: bool = True):
    """Return question cells:
    {index, box(y0,y1,x0,x1), q_img, a_img, hw_answer_img}
    The answer zone is geometric: handwriting to the right of the last
    printed column of the row (the '=' or blank), which is far more robust
    on worksheets than projection valleys.
    """
    rulings = detect_rulings(printed) if use_rulings else []
    rows = projection_rows(printed)
    cells = []
    W = printed.shape[1]
    H = printed.shape[0]
    for (ry0, ry1) in rows:
        row_p_full = printed[ry0:ry1]
        if not _looks_like_question(row_p_full):
            continue
        x0, x1 = _content_span(row_p_full)
        if x1 - x0 < 60:
            continue
        # children write TALLER than the printed line — expand the row band
        # vertically so the answer is never clipped mid-stroke
        y0, y1 = max(0, ry0 - 12), min(H, ry1 + 16)
        row_p_full = printed[y0:y1]
        # the child's answer lies to the RIGHT of the printed '=' — crop the
        # handwriting from the FULL row width, never from the printed span
        row_hw_full = handwriting[y0:y1]
        q_img, ans_hw = answer_zone(row_p_full, row_hw_full)
        cells.append({
            "box": [int(y0), int(y1), int(x0), int(W - 1)],
            "q_img": q_img,
            "a_img": ans_hw,
            "hw_answer_img": ans_hw,
            "rulings": rulings,
        })
    return cells


def _content_span(row_printed: np.ndarray):
    proj = np.count_nonzero(row_printed, axis=0)
    xs = np.nonzero(proj > 0)[0]
    return (int(xs[0]), int(xs[-1]) + 1) if xs.size else (0, row_printed.shape[1])
