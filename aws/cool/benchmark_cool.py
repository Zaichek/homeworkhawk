#!/usr/bin/env python3
"""COOL benchmark — times HomeworkHawk's hot ops and the full-page grade.

COOL (Cloud-Optimized OpenCV Library, AWS Marketplace) accelerates resize,
adaptive gaussian and contour detection — the three hottest operations in the
HomeworkHawk pipeline. This script measures them plus end-to-end page latency
so a COOL build and the stock opencv-python-headless build can be compared on
the same machine with the same inputs.

Protocol (for the Best Use of COOL prize claim):
  1. Run on a Graviton (arm64) instance with the STOCK wheel installed:
        python benchmark_cool.py --label stock --out bench_stock.json
  2. Switch the OpenCV build to COOL (same interpreter, same instance):
        python benchmark_cool.py --label cool --out bench_cool.json
  3. Diff the two JSONs — speedup per op and end-to-end.

Local runs (any arch) work too and are useful as smoke tests; label them so
Graviton numbers are never mixed with desktop numbers.
"""
import argparse
import json
import platform
import time

import cv2
import numpy as np


def build_info_flags():
    info = cv2.getBuildInformation().lower()
    return {
        "arch_target": next((l.split(":")[1].strip() for l in info.splitlines() if "target" in l and "arch" in l), "unknown"),
        "neon": "neon" in info,
        "third_party": [t for t in ("neon", "simd", "openmp", "tbb") if t in info],
    }


def make_synthetic_page(w=1366, h=720):
    """A worksheet-like page: white paper, print rows, handwriting strokes, ruling lines."""
    rng = np.random.default_rng(7)
    page = np.full((h, w), 245, np.uint8)
    for i, y in enumerate(range(90, h - 60, 60)):  # question rows
        cv2.line(page, (60, y + 40), (w - 60, y + 40), 180, 1)
        cv2.putText(page, f"{i + 1}. 23+58=", (60, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 30, 2)
        cv2.putText(page, "81", (420, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 60, 2)
    for _ in range(400):  # paper noise
        x, y = rng.integers(0, w), rng.integers(0, h)
        page[y, x] = rng.integers(120, 230)
    return page


def bench_ops(img, repeats=200):
    """The pipeline's hot ops, individually timed (median of N)."""
    img2 = cv2.GaussianBlur(img, (3, 3), 0)  # warm-up
    del img2
    t = {}

    def clock(name, fn, n=repeats):
        ts = []
        for _ in range(n):
            s = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - s) * 1000)
        t[name] = round(sorted(ts)[n // 2], 4)

    clock("resize_0.5", lambda: cv2.resize(img, None, fx=0.5, fy=0.5))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clock("adaptive_gaussian", lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10))
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    clock("contours", lambda: cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    clock("gaussian_blur3", lambda: cv2.GaussianBlur(img, (3, 3), 0))
    clock("canny", lambda: cv2.Canny(gray, 60, 180))
    return t


def bench_full_pipeline(img, repeats=30):
    """End-to-end rectify+binarize+segment approximation (what handle_photo does per page)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ts = []
    for _ in range(repeats):
        s = time.perf_counter()
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    M = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [1000, 0], [1000, 700], [0, 700]], np.float32))
                    warp = cv2.warpPerspective(gray, M, (1000, 700))
                    bw = cv2.adaptiveThreshold(warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
        ts.append((time.perf_counter() - s) * 1000)
    return {"grade_page_ms_median": round(sorted(ts)[len(ts) // 2], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="stock", help="build label, e.g. stock / cool")
    ap.add_argument("--out", default=None, help="write JSON result here")
    ap.add_argument("--image", default=None, help="optional real worksheet photo (else synthetic)")
    args = ap.parse_args()

    img = cv2.imread(args.image) if args.image else cv2.cvtColor(make_synthetic_page(), cv2.COLOR_GRAY2BGR)
    result = {
        "label": args.label,
        "cv2_version": cv2.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "build_flags": build_info_flags(),
        "image": args.image or "synthetic_1366x720",
        "ops_ms_median": bench_ops(img),
        "pipeline": bench_full_pipeline(img),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
