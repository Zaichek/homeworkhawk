# HomeworkHawk — Technical Report

OpenCV AI Competition 2026 (powered by AWS) · Team Zaichek Labs · Agentic Vision path

## 1. Problem and users

Every evening, parents check children's math homework by hand — slow, error-prone, hard
for grandparents and working parents. Photo-solving apps *answer* exercises (inviting
copying); almost none **grade what the child actually wrote**. HomeworkHawk grades the
child's own handwriting from a single photo, explains each error at the digit level, and
keeps the parent in the loop for anything ambiguous. Primary users: parents of
elementary-school children (we are our own first users); secondary: tutors and
after-school programs batch-checking worksheets.

## 2. Architecture

`docs/architecture.png` — phone photo → quality gate → page rectification →
illumination flattening → print/handwriting separation → question segmentation →
per-digit answer reading → agentic decision loop (re-binarize / re-read / escalate) →
graded overlay + trace. AWS: the identical handler runs on Lambda behind a Function URL
with an OpenCV 5 headless layer; S3 stores per-stage evidence; CloudWatch records
per-stage latency and tool-call counts. The local demo endpoint (`src/app/web.py`) and
the Lambda adapter share `handle_photo` byte-for-byte.

## 3. OpenCV 5 implementation details

- **Quality gate**: `cv2.Laplacian` variance (focus), clipped-white fraction ≥254 in HSV
  (glare — deliberately not "bright", so white paper is not misjudged), dark fraction.
- **Page rectification**: CLAHE → Canny(lo,hi) → morphological close → `findContours` +
  `approxPolyDP` quadrilateral (area ≥25% of frame, convex) → point ordering →
  `getPerspectiveTransform` + `warpPerspective`. All four detection parameters are
  function arguments, because the agent re-invokes detection with looser parameters
  when the first attempt fails.
- **Illumination flattening**: `cv2.divide(gray, GaussianBlur(gray, σ=25))` — removes
  phone-shadow gradients that break global thresholds.
- **Print/handwriting separation**: two-level intensity classification on the flattened
  page — toner print is far darker than pencil. `print_level=70` splits them (tunable:
  the agent lowers it for faded photocopies); an adaptive-threshold `ink` mask catches
  all marks; handwriting = ink minus a 3×3-dilated print mask (kills anti-alias halo).
- **Question segmentation**: horizontal projection of the printed mask
  (`count_nonzero(axis=1)`), ruling lines excluded from the printed x-extent
  computation (run-length > 60% row width), header rows rejected via leading-blob width
  (question numbers are narrow; the CJK title is a wide dense block), rows expanded
  ±12–16 px vertically because children write taller than print, answer zone = full-row
  handwriting to the right of the last printed column (the "=").
- **Answer reading**: connected components (`connectedComponentsWithStats`) → fragment
  merging restricted to small-area strokes only (a thin "1" is a full glyph, not a
  fragment — we learned this the hard way) → aspect-preserving square-pad → fixed
  40×30 grid → cosine similarity against rendered digit-template banks
  (Ink Free / Segoe Print / Comic / Arial at three sizes).
- **Agentic loop**: per-digit mean confidence gates the next action
  (≥0.72 grade · <0.72 re-binarize with Otsu on the flattened row and re-read ·
  still weak → escalate). Every step lands in a `Step` record with tool, params,
  perception, decision.

## 4. Evaluation

Synthetic labeled corpus: 3 worksheets × 6 degradations (flat / perspective-warped /
diagonal shadow / Gaussian blur / faint pencil / half-erased answer), 180 questions,
wrong-answer injection at 30% to test wrong-detection. Generator:
`tests/make_samples.py`; harness: `tests/test_pipeline.py`; full per-run evidence:
`evidence/`.

| Metric | Result |
|---|---|
| Page IoU (perspective, vs generator ground truth) | 0.998 (all 3) |
| Question segmentation recall | 146/180 = 81.1% (flat/shadow/faint/erase ≈ 100%; persp 8–9/10; blur gated) |
| Answer exact-read | 139/149 = 93.3% |
| Grading decision accuracy | 125/146 = 85.6% |
| Gate behavior on blur | 3/3 correctly demanded re-shoot |
| Latency | 0.15–0.30 s per page, CPU only (i7-class laptop, single thread) |

**Failure cases (published in `evidence/`):**
- Perspective warps lose 1–2 edge rows (projection valleys shift under homography).
- The reader is font-biased: it matches against rendered handwriting-like fonts, so
  unusual personal handwriting lowers confidence — by design this routes to escalation
  rather than silent errors, but it caps unattended accuracy.
- Half-erased answers: when the erasure obliterates >60% of a digit, confidence drops
  and the question escalates; when it doesn't, we sometimes read the erased digit
  anyway (pencil ghosts) — a genuinely hard case we show rather than hide.
- Two wrong-answer reads at 0.77 confidence were graded "wrong" correctly; the residual
  85.6% decision accuracy comes from reads that are confidently wrong on messy digits.

**Agentic-loop statistics** (from `evidence/metrics.json` `attempts_hist`): ~78% of
questions grade in one pass; ~22% trigger the Otsu re-analysis tool; of those, roughly
two thirds resolve to a confident verdict and one third escalate to the human.

## 5. AWS deployment

`src/lambda_handler/lambda_function.py` + `aws/README.md`:
Lambda (Python 3.12, arm64 Graviton or x86_64) with an
opencv-python-headless 5.x layer; Function URL returns the same JSON as the demo
(`image_b64` in, graded overlay + trace out); S3 bucket per-stage evidence with
lifecycle expiry; CloudWatch custom metrics per stage. Free-tier sized: the whole
pipeline is <0.3 s CPU, so a 512 MB function stays deep inside free tiers.

## 6. Responsible-use considerations

Children's homework photos: transient processing, evidence objects expire, no faces in
frame, on-device/local by default (the cloud endpoint is optional). The product grades
and explains; it does not solve exercises for the child. Every ambiguous decision is
escalated to the parent — the system is designed to say "I'm not sure" instead of
guessing about a child's work. Digit-level error localization turns "wrong" into
"the 3rd digit should be 8" — feedback that teaches rather than judges.

## 7. Limitations and what we would build next

Multi-line answers, fractions, and drawn figures are out of scope today. Real-world
handwriting diversity needs a small on-device classifier to replace/augment the
template bank (the trace format already records per-digit crops to train it). The AWS
COOL/Graviton path is sketched but not benchmarked here.
