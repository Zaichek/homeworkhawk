"""The grader agent — a perception → decision → action loop.

For every question cell the agent:
  1. PERCEIVES  the handwriting answer mask (OpenCV 5 binarization + cleanup)
  2. MATCHES    it against the expected answer (template NCC confidence)
  3. DECIDES    based on *visual evidence* what to do next:
       conf >= HIGH      → grade now
       MID <= conf < HIGH → re-run OpenCV 5 stages with different parameters
                           (re-binarize / zoom) and re-match  ← vision changes action
       conf < MID         → escalate to the parent (human-in-the-loop)

Every decision is written to a trace so judges can replay the loop.

LLM hook: `explainer` (optional) turns the per-question verdict into a short
parent-friendly sentence. Absent an LLM, deterministic phrasing is used —
the agentic loop itself never depends on the LLM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from ..pipeline import quality, page as page_mod, separate, segment
from . import matcher

HIGH, MID = 0.80, 0.55


@dataclass
class Step:
    step: int
    tool: str
    params: dict
    perception: dict
    decision: str


@dataclass
class QuestionResult:
    index: int
    expected: str
    score: float
    verdict: str            # correct | wrong | unclear | escalated
    attempts: int
    trace: list[Step] = field(default_factory=list)


@dataclass
class RunResult:
    quality: dict
    page_found: bool
    questions: list[QuestionResult]
    total_seconds: float
    stage_images: dict      # name → png bytes are added by the caller


def process_photo(bgr: np.ndarray, answer_key: dict[int, str],
                  explainer=None) -> RunResult:
    """Full agentic run over one photo. answer_key maps question index →
    expected answer (the parent or worksheet bank supplies it)."""
    t0 = time.perf_counter()
    steps_log: list[Step] = [Step(0, "quality_gate", {}, {}, "")]
    q = quality.assess(bgr)
    if q["verdict"] != "accept":
        return RunResult(q, False, [], round(time.perf_counter() - t0, 3),
                         {"reject_reason": q["reasons"]})

    # page detection with one agentic retry (looser Canny) — perception failure
    # changes the parameters of the next OpenCV 5 call
    warped, quad = page_mod.detect_page(bgr)
    attempt = 1
    if warped is None:
        warped, quad = page_mod.detect_page(bgr, canny_lo=40, canny_hi=120, eps=0.03)
        attempt = 2
    if warped is None:
        # fall back to full-frame processing (flat scans)
        warped, quad = bgr, np.float32([[0, 0], [bgr.shape[1], 0],
                                        [bgr.shape[1], bgr.shape[0]], [0, bgr.shape[0]]])
        steps_log.append(Step(1, "page_detect", {"fallback": "full_frame"}, {},
                              "process_full_frame"))
    else:
        steps_log.append(Step(1, "page_detect", {"attempt": attempt}, {},
                              "rectify"))

    # illumination flattening before separation
    flat = separate.deshadow(warped)
    printed, hw, ink = separate.separate(flat)
    steps_log.append(Step(2, "separate", {"method": "adaptive"},
                          {"stroke_width": round(separate.stroke_width(ink), 2)},
                          "segment"))
    cells = segment.segment_questions(warped, printed, hw)

    results: list[QuestionResult] = []
    for ci, cell in enumerate(cells, start=1):
        expected = str(answer_key.get(ci, ""))
        if not expected:
            results.append(QuestionResult(ci, "?", 0.0, "no_key", 1))
            continue
        mask = cell["hw_answer_img"]
        read = matcher.read_answer(mask)
        cmp = matcher.compare(read, expected)
        attempts = 1
        trace: list[Step] = []
        # --- agentic re-analysis: weak per-digit confidence triggers a NEW
        # vision call (Otsu re-binarization of the row) and a re-read. What
        # the camera shows decides what the system does next.
        if cmp["score"] < 0.72:
            y0, y1, x0, x1 = cell["box"]
            row = warped[y0:y1, x0:x1]
            ink2 = separate.binarize(separate.deshadow(row), "otsu")
            read2 = matcher.read_answer(ink2)
            cmp2 = matcher.compare(read2, expected)
            attempts += 1
            if cmp2["score"] > cmp["score"]:
                read, cmp = read2, cmp2
                trace.append(Step(attempts, "rebinarize+re-read",
                                  {"method": "otsu"},
                                  {"score": cmp2["score"]}, "accept_retry"))
            else:
                trace.append(Step(attempts, "rebinarize+re-read",
                                  {"method": "otsu"},
                                  {"score": cmp2["score"]}, "keep_first"))
        verdict = cmp["verdict"]
        note = cmp.get("why", "")
        if explainer:
            note = explainer(expected, cmp["score"], verdict) or note
        results.append(QuestionResult(ci, expected, cmp["score"], verdict,
                                      attempts, trace))
    total = round(time.perf_counter() - t0, 3)
    return RunResult(q, True, results, total, {})


def render_overlay(warped: np.ndarray, results: list[QuestionResult],
                   cells: list[dict]) -> np.ndarray:
    """Graded page: green=correct, amber=unclear, red=escalated, drawn as a
    heat strip beside each question row."""
    out = warped.copy()
    for res, cell in zip(results, cells):
        y0, y1, x0, x1 = cell["box"]
        color = {"correct": (60, 180, 75), "unclear": (0, 200, 255),
                 "escalated": (50, 50, 230), "no_key": (128, 128, 128),
                 "wrong": (50, 50, 230)}[res.verdict]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        cv2.rectangle(out, (x0, y0), (x0 + 54, y0 + 34), color, -1)
        cv2.putText(out, f"{res.score:.2f}", (x0 + 4, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


def run_to_json(r: RunResult) -> str:
    import json
    d = asdict(r)
    d.pop("stage_images", None)
    return json.dumps(d, ensure_ascii=False, indent=2)
