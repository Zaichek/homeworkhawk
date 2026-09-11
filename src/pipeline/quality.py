"""Capture quality gating — decides if a photo frame is worth processing.

HomeworkHawk · OpenCV AI Competition 2026
Stage 0 of the perception pipeline. Pure OpenCV 5, no external deps.
"""
from __future__ import annotations

import cv2
import numpy as np

# Tunable gates (documented in docs/report.md §evaluation)
BLUR_MIN = 60.0        # Laplacian variance below this = too blurry
GLARE_MAX_RATIO = 0.05 # fraction of near-saturated pixels allowed
DARK_MAX_RATIO = 0.60  # fraction of near-black pixels allowed


def blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — classic focus measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def glare_ratio(bgr: np.ndarray) -> float:
    """Fraction of pixels truly clipped to white (specular highlights).
    Bright-but-not-clipped paper must NOT count as glare, so the threshold
    sits at near-saturation (>= 254), not 'bright'."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.count_nonzero(hsv[..., 2] >= 254) / hsv[..., 2].size)


def dark_ratio(gray: np.ndarray) -> float:
    return float(np.count_nonzero(gray < 25) / gray.size)


def assess(bgr: np.ndarray) -> dict:
    """Return metrics + verdict. The agent uses `verdict` to decide whether
    to process the frame or ask the parent to re-shoot (an agentic action
    triggered purely by visual evidence)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = {
        "blur": round(blur_score(gray), 1),
        "glare": round(glare_ratio(bgr), 4),
        "dark": round(dark_ratio(gray), 4),
        "size": list(bgr.shape[:2]),
    }
    reasons = []
    if m["blur"] < BLUR_MIN:
        reasons.append("blurry")
    if m["glare"] > GLARE_MAX_RATIO:
        reasons.append("glare")
    if m["dark"] > DARK_MAX_RATIO:
        reasons.append("too_dark")
    m["verdict"] = "accept" if not reasons else "reshoot"
    m["reasons"] = reasons
    return m


if __name__ == "__main__":
    import sys

    img = cv2.imread(sys.argv[1])
    print(assess(img))
