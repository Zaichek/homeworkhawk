"""HomeworkHawk on Strands Agents SDK — an LLM agent orchestrating OpenCV 5 vision tools.

The deterministic OpenCV pipeline stays exactly as graded in the OpenCV track
(page IoU 0.998 / 93.3% answer read). What Strands adds is the *orchestration
layer*: a Strands Agent decides, from the visual evidence each tool returns,
what happens next — re-shoot, re-read a low-confidence question with different
binarization, or escalate to the parent. Every tool result is grounded
perception; the agent's next tool call is the action.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from strands import Agent, tool

from src.agent import grader
from src.pipeline import quality


@tool
def assess_photo(image_path: str) -> str:
    """Quality-gate a worksheet photo with OpenCV 5 (blur / glare / darkness).

    Returns JSON with a verdict: 'accept' means worth grading, 'reshoot'
    means the parent must take a better photo first.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return json.dumps({"error": "cannot read image"})
    q = quality.assess(bgr)
    return json.dumps(q, ensure_ascii=False)


@tool
def grade_page(image_path: str, answer_key_json: str) -> str:
    """Grade a whole worksheet page: rectification, print/handwriting
    separation, per-question digit reading, confidence-gated verdicts.

    answer_key_json: {"1": "1081", "2": "1093", ...}
    Returns JSON: per-question verdict/score/attempts plus a list of
    question ids that need attention (low confidence -> candidates for
    re_read_question or escalation).
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return json.dumps({"error": "cannot read image"})
    key = {int(k): str(v) for k, v in json.loads(answer_key_json).items()}
    r = grader.process_photo(bgr, key)
    out = {
        "page_found": r.page_found,
        "total_seconds": round(r.total_seconds, 3),
        "questions": [
            {"q": q.index, "expected": q.expected, "score": round(q.score, 3),
             "verdict": q.verdict, "attempts": q.attempts}
            for q in r.questions
        ],
        "needs_attention": [
            q.index for q in r.questions if q.verdict in ("unclear", "escalated")
        ],
    }
    return json.dumps(out, ensure_ascii=False)


@tool
def re_read_question(image_path: str, answer_key_json: str, question_id: int) -> str:
    """Re-read ONE question with a forced Otsu global binarization (different
    OpenCV 5 parameters than the first pass). Use when grade_page returned a
    low score for that question."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        return json.dumps({"error": "cannot read image"})
    key = {int(k): str(v) for k, v in json.loads(answer_key_json).items()}
    r = grader.process_photo(bgr, key, force_otsu_for={question_id})
    for q in r.questions:
        if q.index == question_id:
            return json.dumps({
                "q": q.index, "score": round(q.score, 3),
                "verdict": q.verdict, "attempts": q.attempts,
            }, ensure_ascii=False)
    return json.dumps({"error": "question not found"})


@tool
def escalate_to_parent(question_id: int, reason: str) -> str:
    """Mark a question for the parent to check by hand. The system NEVER
    guesses about a child's work: if two read passes still disagree with the
    key, a human looks at it."""
    return json.dumps({
        "q": question_id, "escalated": True, "reason": reason,
        "message": f"Question {question_id} flagged for the parent: {reason}",
    }, ensure_ascii=False)


@tool
def sheet_math(left: int, right: int, op: str) -> str:
    """Independent arithmetic check (never trust a matcher blindly): op is one of + - * /"""
    import operator
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul}
    if op in ops:
        return str(ops[op](left, right))
    if op == "/":
        return f"{left / right:.4f}"
    return "unknown op"


SYSTEM_PROMPT = """You are HomeworkHawk, a careful grading assistant for parents.

Policy — follow it strictly, tool evidence decides every step:
1. First call assess_photo. If verdict is 'reshoot', STOP and tell the parent
   to retake the photo. Do not grade a bad photo.
2. Otherwise call grade_page once.
3. For every question in needs_attention, call re_read_question once.
4. If the re-read still returns verdict 'unclear' or score < 0.55, call
   escalate_to_parent for that question — never guess on a child's work.
5. If an expected answer looks arithmetically wrong in the key itself, verify
   with sheet_math and mention it.
6. Finish with a short parent-friendly report: how many correct / wrong /
   escalated, and what to check by hand.

Be concise. Numbers only from tool outputs, never invented."""


def build_agent(model_id: str, api_key: str, base_url: str) -> Agent:
    from strands.models.litellm import LiteLLMModel
    model = LiteLLMModel(
        model_id=model_id,
        api_key=api_key,
        client_args={"base_url": base_url, "api_key": api_key},
    )
    return Agent(
        model=model,
        tools=[assess_photo, grade_page, re_read_question, escalate_to_parent, sheet_math],
        system_prompt=SYSTEM_PROMPT,
    )
