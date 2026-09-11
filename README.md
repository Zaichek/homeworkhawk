# 🦅 HomeworkHawk

**One photo of a finished math worksheet → a graded page, in a third of a second, on CPU.**

An **agentic vision homework grader** for parents: point your phone at the worksheet your
child just finished, and HomeworkHawk finds the page, separates the printed questions from
the pencil answers, reads each handwritten answer, and grades it — escalating to you
(the human) whenever the visual evidence is not confident enough.

Built for the **OpenCV AI Competition 2026 (powered by AWS)** — *Agentic Vision* focus path.

---

## Why it is *agentic* (not just a pipeline)

Every stage is a callable OpenCV 5 tool, and a decision policy sits on top:

```
perceive (OpenCV 5) → decide → act (another OpenCV 5 call, or ask the human)
```

Concretely — visual evidence changes what the system does next:

| Perception (OpenCV 5) | Decision | Action |
|---|---|---|
| blur/glare metrics fail gate | frame unusable | **ask the parent to re-shoot** |
| page quadrilateral not found | first attempt failed | **re-run detection with looser Canny** |
| per-digit confidence < 0.72 | answer ambiguous | **re-binarize the row (Otsu) and re-read** |
| still < threshold | genuinely unclear | **escalate that question to the parent** |

A chat that merely describes a fixed vision result does not qualify as agentic — here the
*vision output itself* drives which OpenCV call happens next, and every decision is
recorded in a replayable trace (`evidence/*/result.json`).

## Pipeline (all OpenCV 5, CPU-only)

1. **Quality gate** — Laplacian-variance blur, clipped-white glare, darkness ratios.
2. **Page detection & rectification** — Canny + contour approximation → 4-point
   `getPerspectiveTransform` → `warpPerspective` (IoU **0.998** on warped photos).
3. **Illumination flattening** — background-estimate division (kills phone shadows).
4. **Print / handwriting separation** — two-level intensity classification on the
   flattened page (toner is far darker than pencil) + dilated-print subtraction.
5. **Question segmentation** — horizontal projection profiles, ruling-line exclusion,
   question-number blob detection, vertical expansion for taller handwriting.
6. **Answer reading** — connected-component character blobs, fragment-only merging,
   aspect-preserved fixed-grid cosine matching against rendered digit templates.
7. **Agentic re-analysis & grading loop** — per-digit confidence gates re-binarize /
   re-read / escalate; verdicts per question with digit-level error localization.

## Measured results (18 synthetic worksheets, 180 questions, degraded 6 ways)

| Metric | Result |
|---|---|
| Page-detection IoU (perspective-warped) | **0.998** |
| Question segmentation recall | **81.1%** overall; ~100% on clean/shadow/faint/erased |
| Handwritten answer exact-read | **93.3%** (139/149 reachable questions) |
| Grading decision accuracy | **85.6%** (125/146 judged) |
| Blur detection (gate correctly demands re-shoot) | 3/3 |
| Latency per page | **0.15–0.30 s**, pure CPU |

Failure cases are published, not hidden: `evidence/` contains every run, including
half-erased answers, perspective losses, and the matcher's font bias (honest limitations
in `docs/report.md`).

## Run the demo

```bash
pip install opencv-python==5.* numpy pillow   # that's the whole dependency list

python tests/make_samples.py                 # generate the labeled test set
python tests/test_pipeline.py                # full evaluation + evidence dump
python src/app/web.py                        # then open http://127.0.0.1:8765
```

In the web UI: upload `samples/w0_shadow.png`, the answer key for that sheet loads
automatically — you get the graded page, the per-question verdicts, and the agent's
re-analysis passes.

## AWS deployment

`src/lambda_handler/lambda_function.py` is the cloud twin of the demo endpoint —
same `handle_photo`, behind a Lambda Function URL, with an
`opencv-python-headless` 5.x layer (x86_64 or Graviton). Wiring details and the
S3/CloudWatch evidence layout: `aws/README.md`.

## Repository layout

```
src/pipeline/   quality gate · page rectification · separation · segmentation
src/agent/      digit reader/matcher · the perception→decision→action grader
src/app/        local web endpoint (judge demo)
src/lambda_handler/  AWS Lambda adapter
tests/          synthetic worksheet generator · evaluation harness
samples/        18 labeled worksheets × 6 degradation cases + manifest
evidence/       per-run stage images, overlays, traces, metrics.json
docs/           technical report · architecture diagram
```

## Responsible use

Photos are processed transiently; no faces are required; ambiguous marks are escalated
to a human rather than guessed; the system grades and explains, it does not solve.
See `docs/report.md` §Responsible-use.

## Team

**Zaichek Labs** — one human directing AI-agent workflows; built to scratch our own itch
(the maintainer's household contains one third-grader and two tired parents).
