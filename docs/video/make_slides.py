# -*- coding: utf-8 -*-
"""Generate the demo-video slides (HTML -> headless Edge -> PNG)."""
import subprocess
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = Path(__file__).parent

CSS = """<style>
@page{size:1600px 900px;margin:0}
body{font-family:"Segoe UI",Arial,sans-serif;margin:0;background:#10141f;color:#e8ecf4;
padding:60px 80px;box-sizing:border-box;width:1600px;height:900px}
h1{font-size:52px;margin:0 0 10px;color:#fff}
h2{font-size:38px;margin:0 0 18px;color:#7ea4ff}
p,li{font-size:27px;line-height:1.55;color:#c7d2e8}
.k{color:#7bd88f;font-weight:700}.o{color:#ffb454;font-weight:700}
img{max-width:1380px;max-height:660px;border-radius:12px;border:1px solid #2a3450}
table{font-size:26px;border-collapse:collapse}
td,th{padding:10px 22px;border-bottom:1px solid #2a3450;text-align:left}
th{color:#7ea4ff}
.big{font-size:44px;color:#fff;font-weight:800}
.small{font-size:22px;color:#8fa3c0}
</style>"""

slides = {
    "s1_title": f"""{CSS}<h1>🦅 HomeworkHawk</h1>
<p class=big>One photo → a graded math worksheet, in 0.24 seconds, on CPU.</p>
<p>Team <span class=k>Zaichek Labs</span> · OpenCV AI Competition 2026, powered by AWS · Agentic Vision path</p>
<p class=small>github.com/Zaichek/homeworkhawk</p>""",

    "s2_problem": f"""{CSS}<h2>The problem</h2>
<ul>
<li>Every evening, hundreds of millions of parents check children's homework <span class=o>by hand</span> — slow, error-prone, hardest for grandparents and working parents.</li>
<li>Photo-solving apps <span class=o>answer</span> exercises — inviting copying. Almost none <span class=k>grade what the child actually wrote</span>.</li>
<li>HomeworkHawk reads <span class=k>the child's own pencil</span> from one photo, grades it, and explains every error <span class=k>at the digit level</span>.</li>
</ul>""",

    "s3_demo1": f"""{CSS}<h2>Live demo — the working web endpoint</h2>
<img src="ui_1_upload.png">""",

    "s4_demo2": f"""{CSS}<h2>Result — 0.24 s after upload, pure OpenCV 5 on CPU</h2>
<img src="ui_2_result.png">""",

    "s5_pipeline": f"""{CSS}<h2>What OpenCV 5 actually does</h2>
<table>
<tr><th>Stage</th><th>OpenCV 5 technique</th></tr>
<tr><td>Quality gate</td><td>Laplacian variance · clipped-white glare — asks for a <span class=k>re-shoot</span> when the photo is unusable</td></tr>
<tr><td>Page rectification</td><td>Contours → approxPolyDP quad → perspective warp — <span class=k>IoU 0.998</span></td></tr>
<tr><td>Flatten + separate</td><td>Background-division deshadow · two-level intensity: toner ≪ pencil</td></tr>
<tr><td>Question segmentation</td><td>Projection profiles · ruling exclusion · answer zone right of "="</td></tr>
<tr><td>Answer reading</td><td>Connected-component glyphs · fixed-grid cosine vs digit templates — <span class=k>93.3%</span> exact read</td></tr>
</table>""",

    "s6_agent": f"""{CSS}<h2>Why it is <span class=k>agentic</span> — vision changes the next action</h2>
<ul>
<li>Blur or glare in the photo → <span class=k>ask the parent to re-shoot</span></li>
<li>Page quadrilateral not found → <span class=k>re-run detection with looser parameters</span></li>
<li>Per-digit confidence below 0.72 → <span class=k>re-binarize the row (Otsu) and re-read</span> — a second OpenCV call decided by what the first one saw</li>
<li>Still uncertain → <span class=k>escalate that question to the human</span> — the system says "not sure" instead of guessing about a child's work</li>
</ul>
<p class=small>Every decision lands in a replayable trace: tool, parameters, perception, decision.</p>""",

    "s7_results": f"""{CSS}<h2>Measured on 18 labeled worksheets × 6 degradations</h2>
<table>
<tr><th>Metric</th><th>Result</th></tr>
<tr><td>Page-detection IoU (perspective-warped)</td><td><span class=k>0.998</span></td></tr>
<tr><td>Question segmentation (clean · shadow · faint · erased)</td><td><span class=k>≈ 100%</span> (overall 81%, incl. warped edges)</td></tr>
<tr><td>Handwritten answer exact-read</td><td><span class=k>93.3%</span></td></tr>
<tr><td>Grading decision accuracy</td><td><span class=k>85.6%</span></td></tr>
<tr><td>Blur photos correctly rejected by the gate</td><td><span class=k>3 / 3</span></td></tr>
<tr><td>Latency per page</td><td><span class=k>0.15 – 0.30 s</span>, CPU only</td></tr>
</table>""",

    "s8_arch": f"""{CSS}<h2>Architecture — same handler on localhost and AWS Lambda</h2>
<img src="../architecture.png">""",

    "s9_limits": f"""{CSS}<h2>Published failure cases &amp; limits</h2>
<ul>
<li>Perspective warps can lose 1–2 edge rows — we show it, not hide it.</li>
<li>The reader is <span class=o>font-biased</span>: unusual personal handwriting lowers confidence — by design that <span class=k>routes to human escalation</span>, never silent errors.</li>
<li>Half-erased digits: pencil ghosts sometimes still read — a genuinely hard case in the evidence gallery.</li>
<li>Responsible use: transient processing, no faces, evidence expires, human approval gates every ambiguous verdict.</li>
</ul>""",

    "s10_close": f"""{CSS}<h1>HomeworkHawk</h1>
<p class=big>Perception → decision → action, in 0.24 seconds.</p>
<p>Everything open: code, labeled test set, per-run evidence, traces.</p>
<p class=big>github.com/Zaichek/homeworkhawk</p>
<p class=small>Team Zaichek Labs — one human + agent workflows, for a third-grader we know. Thank you!</p>""",
}

for name, body in slides.items():
    html = f"<!DOCTYPE html><html><head><meta charset=utf-8>{body}</body></html>"
    (OUT / f"{name}.html").write_text(html, encoding="utf-8")
    png = OUT / f"{name}.png"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--window-size=1600,900",
                    f"--screenshot={png}", (OUT / f"{name}.html").as_uri()],
                   capture_output=True, timeout=60)
    print(name, "ok" if png.exists() else "FAILED")
