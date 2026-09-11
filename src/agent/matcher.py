"""Answer reading & matching.

Strategy: elementary math answers are short digit strings, so we READ them:
  1. split the handwriting mask into character blobs (connected components,
     left-to-right ordering — a classic OpenCV text-line technique)
  2. classify each blob against rendered digit templates (0-9) with
     normalized cross-correlation over a small scale pyramid
  3. compare the read string with the expected answer

Per-digit confidence (not just global similarity) is what the agent acts on:
one ambiguous digit escalates that question, a confident misread marks it
wrong *and names the digit position* — the explanation parents actually want.

Rendered-template reading is font-biased by nature; we report this openly in
the technical report and the agent's escalation path covers the gap.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/inkfree.ttf",
    "C:/Windows/Fonts/segoepr.ttf",
    "C:/Windows/Fonts/Comic.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
DIGITS = "0123456789"
_cache: dict[str, list[np.ndarray]] = {}


def _render(text: str) -> list[np.ndarray]:
    if text in _cache:
        return _cache[text]
    masks = []
    for path in FONT_CANDIDATES:
        for size in (44, 56, 70):
            try:
                f = ImageFont.truetype(path, size)
            except OSError:
                continue
            img = Image.new("L", (80 * max(len(text), 1) + 40, 130), 0)
            ImageDraw.Draw(img).text((20, 15), text, fill=255, font=f)
            m = np.array(img)
            ys, xs = np.nonzero(m)
            if xs.size:
                masks.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    _cache[text] = masks
    return masks


def _ncc_best(blob: np.ndarray, templates: list[np.ndarray]) -> float:
    """Best TM_CCORR_NORMED of blob vs templates at a few scales."""
    best = -1.0
    h0 = blob.shape[0]
    if h0 < 6:
        return 0.0
    for t in templates:
        for s in (0.8, 1.0, 1.2):
            h = max(12, int(h0 * s))
            tw = max(2, int(t.shape[1] * h / t.shape[0]))
            tt = cv2.resize(t, (tw, h), interpolation=cv2.INTER_AREA)
            if tt.shape[0] >= h0 and tt.shape[1] >= blob.shape[1]:
                canvas = np.zeros((tt.shape[0] + h0, tt.shape[1] + blob.shape[1]),
                                  np.float32)
                canvas[:h0, :blob.shape[1]] = blob
                res = cv2.matchTemplate(canvas, tt.astype(np.float32),
                                        cv2.TM_CCORR_NORMED)
                best = max(best, float(res.max()))
    return best


GRID = (40, 30)  # (w, h) — fixed comparison grid, kills scale search


def _grid(mask: np.ndarray) -> np.ndarray:
    """Aspect-preserving: pad to square, then resize to the fixed grid."""
    m = (mask > 0).astype(np.uint8)
    if m.size == 0:
        return np.zeros(GRID[0] * GRID[1], np.float32)
    h, w = m.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = m
    g = cv2.resize(sq, GRID, interpolation=cv2.INTER_AREA)
    v = g.astype(np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _grid_bank() -> dict[str, np.ndarray]:
    bank = {}
    for d in DIGITS:
        bank[d] = np.stack([_grid(t) for t in _render(d)])
    return bank


_BANK: dict[str, np.ndarray] | None = None


def _digit_score(blob: np.ndarray) -> tuple[str, float]:
    """Fixed-grid cosine similarity against every digit template bank."""
    global _BANK
    if _BANK is None:
        _BANK = _grid_bank()
    v = _grid(blob)
    best_d, best_s = "?", -1.0
    for d, bank in _BANK.items():
        s = float(np.max(bank @ v))
        if s > best_s:
            best_d, best_s = d, s
    return best_d, round(best_s, 3)


def char_blobs(mask: np.ndarray, min_area=20, max_w_ratio=2.4):
    """Ordered character blobs of an answer line. Pencil strokes often break
    into 2-3 fragments under binarization, so fragments merge when they are
    horizontally close (<12px) and vertically near-overlapping."""
    m8 = np.ascontiguousarray((mask > 0).astype(np.uint8))
    if m8.size == 0:
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area or h < 12 or h > 130:
            continue
        boxes.append([x, y, w, h, a])
    boxes.sort(key=lambda b: b[0])
    merged = []
    for b in boxes:
        if merged:
            px, py, pw, ph, pa = merged[-1]
            gap_x = b[0] - (px + pw)
            v_overlap = not (b[1] > py + ph + 6 or b[1] + b[3] < py - 6)
            # merge ONLY true fragments (tiny area — stroke breaks), never two
            # full glyph strokes however thin (a handwritten '1' is narrow but
            # has plenty of stroke pixels)
            frag = (b[4] < 60 or pa < 60)
            if gap_x < 8 and v_overlap and frag:
                nx = min(px, b[0]); ny = min(py, b[1])
                merged[-1] = [nx, ny, max(px + pw, b[0] + b[2]) - nx,
                              max(py + ph, b[1] + b[3]) - ny, pa + b[4]]
                continue
        merged.append(list(b))
    if merged:
        med_w = np.median([b[2] for b in merged])
        merged = [b for b in merged if b[2] <= max_w_ratio * med_w + 10]
    return [b[:4] for b in merged]


def read_answer(mask: np.ndarray) -> dict:
    """Read the digit string from a handwriting mask.
    Returns {text, per_digit: [{digit, score}], confidence}."""
    blobs = char_blobs(mask)
    if not blobs:
        return {"text": "", "per_digit": [], "confidence": 0.0}
    per = []
    for (x, y, w, h) in blobs:
        blob = (mask[y:y + h, x:x + w] > 0).astype(np.uint8) * 255
        d, s = _digit_score(blob)
        per.append({"digit": d, "score": s})
    text = "".join(p["digit"] for p in per)
    conf = float(np.mean([p["score"] for p in per]))
    return {"text": text, "per_digit": per, "confidence": round(conf, 3)}


def compare(read: dict, expected: str) -> dict:
    """Turn a read into a verdict proposal the agent can act on."""
    if not read["per_digit"]:
        return {"verdict": "unclear", "why": "no_handwriting", "score": 0.0}
    r, e = read["text"], str(expected)
    digit_conf = read["confidence"]
    if r == e:
        verdict = "correct" if digit_conf >= 0.72 else "unclear"
    else:
        # confidently different read = confidently wrong answer
        verdict = "wrong" if digit_conf >= 0.72 else "unclear"
        pos = next((i for i in range(min(len(r), len(e))) if r[i] != e[i]),
                   min(len(r), len(e)))
        return {"verdict": verdict, "why": f"digit_{pos+1}", "read": r,
                "expected": e, "score": digit_conf}
    return {"verdict": verdict, "read": r, "expected": e, "score": digit_conf}
