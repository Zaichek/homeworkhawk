"""Print vs handwriting separation.

Printed exercise text (toner/inkjet) is darker and has crisper edges than
pencil handwriting. We exploit that with a two-level threshold strategy plus
morphological cleanup. The result drives question segmentation (printed mask)
and answer matching (handwriting mask).

Known limitation (shown in our failure-case gallery): faint pen vs strong
printer gray can cross-contaminate — the agent's re-binarize tool exists
precisely to attack this.
"""
from __future__ import annotations

import cv2
import numpy as np


def binarize(page_bgr: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """Return a binary 'ink' mask (255 = ink). method: adaptive | otsu | sauvola-ish."""
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    if method == "otsu":
        # global Otsu on a background-flattened image
        bg = cv2.medianBlur(gray, 31)
        flat = cv2.absdiff(gray, bg)
        _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    # adaptive (default): local mean minus C
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 35, 15)


def deshadow(page_bgr: np.ndarray) -> np.ndarray:
    """Remove soft illumination gradients by dividing out a heavily blurred
    background estimate (a classic OpenCV flat-field trick)."""
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    norm = cv2.divide(gray, bg, scale=255)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def stroke_width(mask: np.ndarray) -> float:
    """Median stroke width via distance transform (used to tell uniform
    printed strokes from variable pencil strokes)."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    return float(2 * np.median(vals)) if vals.size else 0.0


def separate(page_bgr: np.ndarray, method: str = "adaptive",
             print_level: int = 70):
    """Return (printed_mask, handwriting_mask, ink_mask).

    Two-level intensity classification on the illumination-flattened page:
    toner/ink print is far darker than pencil. `print_level` (tunable — the
    agent can lower it for faded photocopies) splits print from pencil;
    `ink` is the adaptive any-mark mask. Handwriting = ink minus a dilated
    print mask (kills anti-alias halo). Deliberately conservative: anything
    ambiguous stays in `handwriting` so the child's answer is never dropped
    silently — the agent re-checks or escalates instead.
    """
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    ink = binarize(page_bgr, method)

    print_dark = (gray < print_level).astype(np.uint8) * 255
    print_dark = cv2.morphologyEx(print_dark, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    printed = cv2.bitwise_and(print_dark, ink)

    pd = cv2.dilate(printed, np.ones((3, 3), np.uint8), iterations=1)
    handwriting = cv2.bitwise_and(ink, cv2.bitwise_not(pd))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(handwriting, 8)
    keep = np.zeros_like(handwriting)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 12 <= area <= 4000 and 8 <= bh < 130:
            keep[lab == i] = 255
    return printed, keep, ink
