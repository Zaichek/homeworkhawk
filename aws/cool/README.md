# COOL — Cloud-Optimized OpenCV Library integration (Best Use of COOL prize track)

[COOL](https://opencv.org/cool/) is OpenCV's high-performance build for **AWS
Graviton / ARM**, distributed through the [AWS
Marketplace](https://aws.amazon.com/marketplace/seller-profile?id=seller-q3au5rsqt2mq6).
It accelerates exactly the three operations HomeworkHawk's grading pipeline
runs hottest per page: **resize, adaptive gaussian thresholding, and contour
detection**.

This folder contains the benchmark and the deployment protocol for the
**Best Use of COOL** prize claim (OpenCV AI Competition 2026).

## Files

- `benchmark_cool.py` — measures the pipeline's hot ops (median ms of 200
  repeats) plus end-to-end rectify+binarize+segment latency. Auto-records the
  cv2 build info (arch/NEON/SIMD flags) so a COOL run and a stock run are
  self-identifying.
- `bench_stock_local_x86.json` — smoke-test baseline (desktop x86, stock
  OpenCV 5.0.0). **Not** a Graviton number; kept to prove the harness works.

## Measurement protocol (Graviton)

```bash
# 1. stock baseline — Graviton instance (e.g. c7g.large), stock wheel
pip install opencv-python-headless==5.0.0.93 numpy
python benchmark_cool.py --label stock-graviton --out bench_stock_graviton.json

# 2. COOL build — same instance, install COOL from the AWS Marketplace
#    artifact (seller profile: seller-q3au5rsqt2mq6), same interpreter
python benchmark_cool.py --label cool-graviton --out bench_cool.json

# 3. diff → per-op speedup + end-to-end speedup
```

`[FILL: Graviton stock vs COOL table — pending AWS account run]`

## Why HomeworkHawk is a natural COOL showcase

Every photographed page goes through: quality-gate resize → illumination
flatten → adaptive gaussian binarization (twice: page-level and per-question
re-read) → contour detection for the page quad → warp → row segmentation.
On a 1366×720 page the local smoke test already shows **adaptive gaussian is
the single hottest op (≈7.2 ms of the budget)** — precisely COOL's headline
acceleration target. At Lambda scale (classroom × 30 worksheets/night), a
per-page saving multiplies into real cost savings on the cheapest Graviton
tier.

## Lambda deployment (arm64)

The handler in `../README.md` already builds an arm64 layer. The COOL path
swaps the layer contents: install the Marketplace COOL artifact into
`layer/python/` instead of the PyPI wheel — the handler code is unchanged
(COOL is a drop-in `cv2` build). Benchmark inside Lambda with the same
script via a `--label cool-lambda` run logged to CloudWatch.

## Prize-claim narrative skeleton

1. Same pipeline, two builds, one script — numbers are reproducible by judges
   with the exact commands above.
2. COOL accelerates our top-3 ops by [FILL: xN]; end-to-end [FILL: xN]
   (0.24 s/page desktop → [FILL] ms on Graviton+COOL).
3. Cost story: cheapest Graviton tier, free-tier sized; per-classroom nightly
   batch cost drops from [FILL] to [FILL].
