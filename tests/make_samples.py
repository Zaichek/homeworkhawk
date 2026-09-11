"""Synthetic worksheet generator with ground truth + controlled degradations.

Why synthetic: perfectly labeled data for measurable evaluation (page IoU,
segmentation accuracy, grading accuracy), plus reproducible hard cases
(shadow / perspective / blur / faint pencil / half-erased) that are hard to
collect consistently from a real third-grader at 11pm.

Degradations are applied with OpenCV 5 itself (homographies, Gaussian blur,
alpha gradients) — the same library the pipeline uses.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples"
PRINT_FONT = "C:/Windows/Fonts/calibri.ttf"
HAND_FONTS = ["C:/Windows/Fonts/inkfree.ttf", "C:/Windows/Fonts/segoepr.ttf",
              "C:/Windows/Fonts/Comic.ttf"]
PAPER = (252, 250, 244)   # warm paper, BGR-ish handled as RGB in PIL
INK = (28, 30, 36)
PENCIL = (95, 99, 108)

random.seed(2026)


def gen_questions(n=10):
    qs, answers = [], []
    for i in range(n):
        kind = random.choice(["add", "sub", "mul", "fill"])
        if kind == "add":
            a, b = random.randint(120, 899), random.randint(120, 899)
            qs.append(f"{a} + {b} ="); answers.append(str(a + b))
        elif kind == "sub":
            a, b = random.randint(300, 999), random.randint(100, 299)
            qs.append(f"{a} - {b} ="); answers.append(str(a - b))
        elif kind == "mul":
            a, b = random.randint(3, 9), random.randint(4, 9)
            qs.append(f"{a} × {b} ="); answers.append(str(a * b))
        else:
            a = random.randint(2, 9); qs.append(f"( ) × {a} = 63" if a == 7 else f"{a} × ( ) = {a * random.randint(4, 9)}")
            answers.append(str(63 // a) if "63" in qs[-1] else "")
            if not answers[-1]:
                # regenerate simple: a × ( ) = c
                c = a * random.randint(4, 9)
                qs[-1] = f"{a} × ( ) = {c}"
                answers[-1] = str(c // a)
    return qs, answers


def render_worksheet(qs, answers, wrong_rate=0.3):
    W, H = 1240, 1754  # A4 @150dpi
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    f_head = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 44)
    f_print = ImageFont.truetype(PRINT_FONT, 40)
    f_hw = [ImageFont.truetype(p, 44) for p in HAND_FONTS if Path(p).exists()]
    d.text((90, 70), "数学作业  三年级(2)班", fill=INK, font=f_head)

    y = 170
    line_gap = 150
    ruling_ys = []
    truth = []
    for i, (q, ans) in enumerate(zip(qs, answers), start=1):
        ruling_ys.append(y + 96)
        d.line((70, y + 96, W - 70, y + 96), fill=(150, 152, 158), width=2)
        d.text((90, y), f"{i}. {q}", fill=INK, font=f_print)
        # the child writes the answer after '=' — sometimes wrong
        write = ans if random.random() > wrong_rate else _perturb(ans)
        xf = 90 + int(f_print.getlength(f"{i}. {q}")) + 18
        d.text((xf, y - 6), write, fill=PENCIL, font=random.choice(f_hw))
        truth.append({"q": i, "expected": ans, "written": write,
                      "is_correct": write == ans})
        y += line_gap
    return np.array(img), truth, ruling_ys


def _perturb(ans: str) -> str:
    digits = list(ans)
    i = random.randrange(len(digits))
    digits[i] = str((int(digits[i]) + random.choice([1, 2, 3])) % 10)
    s = "".join(digits)
    return s if s != ans else _perturb(ans)


# ---------------- degradations (OpenCV) ----------------

def degrade_perspective(bgr, max_tilt=0.06):
    h, w = bgr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx, dy = w * max_tilt, h * max_tilt
    dst = np.float32([[dx * .4, dy], [w - dx * .2, dy * .5],
                      [w - dx * .9, h - dy * .4], [dx * .8, h - dy * .8]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(bgr, M, (w, h), borderValue=(200, 205, 210))
    return out, dst  # dst = ground-truth quad


def degrade_shadow(bgr, direction="diag"):
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if direction == "diag":
        g = 1 - 0.38 * ((xx / w + yy / h) / 2)
    else:
        g = 1 - 0.38 * (xx / w)
    return (bgr * g[..., None]).astype(np.uint8)


def degrade_blur(bgr, k=9):
    return cv2.GaussianBlur(bgr, (k, k), 0)


def degrade_faint(bgr):
    """Faint pencil: lift pencil-gray strokes toward paper, keep print dark."""
    out = bgr.copy()
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    zone = ((g > 78) & (g < 145))  # pencil gray range; print stays ~31
    lifted = cv2.convertScaleAbs(out, alpha=0.62, beta=110)
    out[zone] = lifted[zone]
    return out


def degrade_erase(bgr, truth_ys, xf_guess=560):
    """Simulate a half-erased answer: paint paper-colored strokes over part
    of one answer area."""
    out = bgr.copy()
    y = truth_ys[0]  # first question
    for x in range(xf_guess, xf_guess + 90):
        cv2.line(out, (x, y - 26), (x + 4, y + 16), (250, 248, 242), 6)
    return out


def make_dataset():
    OUT.mkdir(exist_ok=True)
    manifest = []
    for batch in range(3):
        qs, answers = gen_questions(10)
        base, truth, ruling_ys = render_worksheet(qs, answers)
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

        cases = {
            "flat": (bgr, None),
            "persp": degrade_perspective(bgr),
            "shadow": (degrade_shadow(bgr), None),
            "blur": (degrade_blur(bgr), None),
            "faint": (degrade_faint(bgr), None),
            "erase": (degrade_erase(bgr, ruling_ys), None),
        }
        for name, (img, quad) in cases.items():
            fname = f"w{batch}_{name}.png"
            cv2.imwrite(str(OUT / fname), img)
            manifest.append({
                "file": fname, "truth": truth,
                "quad": quad.tolist() if quad is not None else None,
                "case": name,
            })
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"generated {len(manifest)} samples -> {OUT}")


if __name__ == "__main__":
    make_dataset()
