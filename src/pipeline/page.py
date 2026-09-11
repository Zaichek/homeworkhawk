"""Page detection & perspective rectification.

Finds the worksheet quadrilateral and warps it into an upright, flat page.
If detection fails the caller (agent) may retry with looser parameters —
`detect_page` takes them explicitly so re-analysis is a tool call, not magic.
"""
from __future__ import annotations

import cv2
import numpy as np

WORK_MAX = 1400  # working resolution for detection


def _order(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _quad_candidates(edges: np.ndarray, area_ref: float, eps: float):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 0.25 * area_ref:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append((a, approx.reshape(4, 2)))
    quads.sort(key=lambda t: -t[0])
    return quads


def detect_page(bgr: np.ndarray, canny_lo: int = 60, canny_hi: int = 180,
                eps: float = 0.02, morph: int = 3):
    """Return (warped_page_bgr, quad_in_original_coords) or (None, None).

    Parameterized on purpose: the grader agent re-invokes this with different
    (canny_lo, canny_hi, eps, morph) when its first attempt fails — an
    OpenCV 5 stage used as an agentic tool.
    """
    h, w = bgr.shape[:2]
    scale = WORK_MAX / max(h, w)
    work = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    edges = cv2.Canny(gray, canny_lo, canny_hi)
    if morph:
        k = np.ones((morph, morph), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    area_ref = work.shape[0] * work.shape[1]
    quads = _quad_candidates(edges, area_ref, eps)
    if not quads:
        return None, None

    quad = _order(quads[0][1].astype(np.float32))
    # if the photo is already a flat scan (quad ≈ full frame), still warp for
    # a canonical orientation & size
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return None, None
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(work, M, (width, height))
    # map quad back to original image coordinates
    inv = 1.0 / scale if scale < 1 else 1.0
    return warped, (quad * inv).astype(np.float32)


def rectify_with_quad(bgr: np.ndarray, quad: np.ndarray):
    """Re-run the warp on the full-resolution image using a known quad."""
    tl, tr, br, bl = [np.float32(p) for p in quad]
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(bgr, M, (width, height))


def iou_quad(qa: np.ndarray, qb: np.ndarray, shape) -> float:
    """IoU between two quads drawn on a mask of `shape` — used by the
    evaluation harness against hand-labeled ground truth."""
    def mask(q):
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [np.int32(q)], 255)
        return m
    ma, mb = mask(qa), mask(qb)
    inter = np.count_nonzero(ma & mb)
    union = np.count_nonzero(ma | mb)
    return inter / union if union else 0.0
