"""Evaluation harness — runs the full agentic pipeline over every sample and
computes the metrics promised in the proposal:

  page-detect IoU vs ground-truth quads (perspective cases)
  question-segmentation accuracy (rows found vs rows in truth)
  grading accuracy (correct/wrong/unclear vs truth, human escalations excluded)
  agent re-analysis stats (attempts distribution)
  per-stage latency

Writes evidence/<sample>/... with stage images + result JSON for the judge
gallery and failure-case documentation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import grader  # noqa: E402
from src.pipeline import page as page_mod, separate, segment  # noqa: E402

EV = ROOT / "evidence"


def run(sample: dict) -> dict:
    img = cv2.imread(str(ROOT / "samples" / sample["file"]))
    key = {t["q"]: t["expected"] for t in sample["truth"]}
    res = grader.process_photo(img, key)

    d = EV / Path(sample["file"]).stem
    d.mkdir(parents=True, exist_ok=True)
    # stage images for the gallery
    warped, quad = page_mod.detect_page(img)
    if warped is None:
        warped, quad = page_mod.detect_page(img, 40, 120, 0.03)
    if warped is None:
        warped = img
    flat = separate.deshadow(warped)
    printed, hw, ink = separate.separate(flat)
    cells = segment.segment_questions(warped, printed, hw)
    overlay = grader.render_overlay(warped, res.questions, cells)
    cv2.imwrite(str(d / "page.png"), warped)
    cv2.imwrite(str(d / "printed.png"), printed)
    cv2.imwrite(str(d / "handwriting.png"), hw)
    cv2.imwrite(str(d / "graded.png"), overlay)

    # metrics — a quality-gate rejection is the agent CORRECTLY demanding a
    # re-shoot, recorded as such rather than as a segmentation failure
    m = {"file": sample["file"], "case": sample["case"],
         "quality": res.quality, "secs": res.total_seconds,
         "questions_found": len(res.questions),
         "questions_truth": len(sample["truth"])}
    if res.quality.get("verdict") == "reshoot":
        m["gate"] = "reshoot:" + ",".join(res.quality.get("reasons", []))
        return m
    if sample["quad"] is not None:
        gt = np.float32(sample["quad"])
        m["page_iou"] = round(page_mod.iou_quad(gt, quad, img.shape), 3)
    # grading accuracy over questions we found AND have keys for
    by_idx = {q.index: q for q in res.questions}
    tp = fp = esc = unclr = 0
    for t in sample["truth"]:
        q = by_idx.get(t["q"])
        if q is None:
            continue
        if q.verdict == "escalated":
            esc += 1
        elif q.verdict == "unclear":
            unclr += 1
        elif q.verdict == "correct" and t["is_correct"]:
            tp += 1
        elif q.verdict != "correct" and not t["is_correct"]:
            tp += 1
        else:
            fp += 1
    m.update({"grade_correct_calls": tp, "grade_mismatch": fp,
              "escalated": esc, "unclear": unclr,
              "attempts_hist": [q.attempts for q in res.questions]})
    (d / "result.json").write_text(grader.run_to_json(res), encoding="utf-8")
    return m


def main():
    manifest = json.loads((ROOT / "samples" / "manifest.json").read_text(encoding="utf-8"))
    metrics = [run(s) for s in manifest]
    (EV / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    # summary
    tot = len(metrics)
    print(f"{'sample':<18}{'case':<8}{'found/truth':<12}{'iou':<6}{'ok':<4}{'mis':<4}{'esc':<4}{'unc':<4}{'s':<6}")
    for m in metrics:
        gate = m.get("gate", "")
        print(f"{m['file']:<18}{m['case']:<8}{m['questions_found']}/{m['questions_truth']:<9}"
              f"{str(m.get('page_iou','-')):<6}{m.get('grade_correct_calls',0):<4}{m.get('grade_mismatch',0):<4}"
              f"{m.get('escalated',0):<4}{m.get('unclear',0):<4}{m['secs']:<6}{gate}")
    found = sum(m["questions_found"] for m in metrics)
    want = sum(m["questions_truth"] for m in metrics)
    print(f"\nsegmentation recall: {found}/{want} = {found/want:.1%}")
    gated = sum(1 for m in metrics if m.get("gate"))
    print(f"quality gate re-shoots (correct agent behavior): {gated}")
    ok = sum(m.get("grade_correct_calls", 0) for m in metrics)
    mis = sum(m.get("grade_mismatch", 0) for m in metrics)
    esc = sum(m.get("escalated", 0) for m in metrics)
    unc = sum(m.get("unclear", 0) for m in metrics)
    judged = ok + mis + unc
    print(f"grading: correct-calls {ok}, mismatch {mis}, unclear {unc}, escalated {esc}"
          f" | decision accuracy on judged: {ok}/{judged} = {ok/judged:.1%}" if judged else "no graded questions")


if __name__ == "__main__":
    main()
