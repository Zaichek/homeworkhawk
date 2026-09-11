"""HomeworkHawk local web endpoint — the judge-facing demo.

Zero-dependency (stdlib + project modules): serves a single page with
photo upload, runs the full agentic pipeline, and returns the graded page
overlay + the agent's decision trace. The same `handle_photo` function is
what the AWS Lambda adapter calls, so the demo endpoint and the cloud
endpoint are behaviorally identical.

Run:  python src/app/web.py [port=8765]
Open: http://127.0.0.1:8765
"""
from __future__ import annotations

import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent import grader  # noqa: E402
from src.pipeline import page as page_mod, separate, segment  # noqa: E402


def handle_photo(bgr: np.ndarray, answer_key: dict) -> dict:
    """One photo in, everything the UI needs out. Shared with Lambda."""
    answer_key = {int(k): str(v) for k, v in (answer_key or {}).items()}
    res = grader.process_photo(bgr, answer_key)
    warped, quad = page_mod.detect_page(bgr)
    if warped is None:
        warped, quad = page_mod.detect_page(bgr, 40, 120, 0.03)
    if warped is None:
        warped = bgr
    flat = separate.deshadow(warped)
    printed, hw, ink = separate.separate(flat)
    cells = segment.segment_questions(warped, printed, hw)
    overlay = grader.render_overlay(warped, res.questions, cells)

    def b64(img):
        ok, buf = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(buf).decode()

    return {
        "quality": res.quality,
        "seconds": res.total_seconds,
        "questions": [
            {"i": q.index, "expected": q.expected, "score": q.score,
             "verdict": q.verdict, "attempts": q.attempts}
            for q in res.questions
        ],
        "images": {
            "page": b64(warped),
            "printed": b64(printed),
            "handwriting": b64(hw),
            "graded": b64(overlay),
        },
    }


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>HomeworkHawk — agentic homework grader</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#10141f;color:#e8ecf4;margin:0;padding:24px}
 h1{margin:0 0 4px}.sub{color:#8fa3c0;margin-bottom:18px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .card{background:#1a2133;border-radius:10px;padding:14px}
 img{max-width:100%;border-radius:6px;background:#fff}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{padding:5px 8px;border-bottom:1px solid #2a3450;text-align:left}
 .ok{color:#7bd88f}.unc{color:#ffd166}.esc{color:#ff6b6b}.wr{color:#ff6b6b}
 #drop{border:2px dashed #3b4a6b;border-radius:10px;padding:26px;text-align:center;color:#8fa3c0}
 button{background:#3d6df2;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
 #key{width:100%;box-sizing:border-box;padding:8px;border-radius:6px;border:1px solid #2a3450;background:#0d1120;color:#e8ecf4}
 .tag{display:inline-block;padding:2px 8px;border-radius:99px;background:#24304d;font-size:12px;margin-right:6px}
</style></head><body>
<h1>🦅 HomeworkHawk</h1>
<div class=sub>OpenCV 5 + agentic re-analysis · one photo → graded page ·
 <span class=tag>perception</span><span class=tag>decision</span><span class=tag>action</span></div>
<div class=grid>
 <div class=card>
  <b>1 · Upload or drop a worksheet photo</b>
  <div id=drop>… or pick from <code>samples/</code>: try w0_flat.png / w0_shadow.png / w0_persp.png</div>
  <input type=file id=f accept="image/*" style="margin:12px 0">
  <b>2 · Answer key</b> <span style="color:#8fa3c0">(question number → expected answer, JSON)</span>
  <textarea id=key rows=4 style="margin:8px 0"></textarea>
  <button onclick=go()>Grade it</button>
  <div id=meta style="margin-top:10px;color:#8fa3c0"></div>
 </div>
 <div class=card><b>Agent decisions</b><table id=t></table></div>
 <div class=card><b>Graded page</b><br><img id=graded></div>
 <div class=card><b>Stage evidence</b> — handwriting mask (what the matcher actually saw)<br><img id=hw></div>
</div>
<script>
const DEFAULT_KEY = fetch('/default_key').then(r=>r.json()).then(d=>{document.getElementById('key').value=JSON.stringify(d)});
async function go(){
 const f=document.getElementById('f').files[0];
 if(!f){alert('pick a photo (samples/ folder of the repo has ready ones)');return}
 const key=JSON.parse(document.getElementById('key').value||'{}');
 document.getElementById('meta').textContent='running pipeline…';
 const fd=new FormData(); fd.append('photo',f); fd.append('key',JSON.stringify(key));
 const r=await fetch('/grade',{method:'POST',body:fd}); const j=await r.json();
 document.getElementById('meta').innerHTML=`gate: <b>${j.quality.verdict}</b> (${j.quality.reasons||'clean'}) · ${j.seconds}s · pure-OpenCV CPU`;
 document.getElementById('graded').src=j.images.graded;
 document.getElementById('hw').src=j.images.handwriting;
 let h='<tr><th>#</th><th>expected</th><th>conf</th><th>verdict</th><th>re-analyzed</th></tr>';
 for(const q of j.questions) h+=`<tr><td>${q.i}</td><td>${q.expected}</td><td>${q.score.toFixed(2)}</td><td class="${q.verdict}">${q.verdict}</td><td>${q.attempts>1?'✓ '+q.attempts+' passes':'—'}</td></tr>`;
 document.getElementById('t').innerHTML=h;
}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/default_key":
            mf = json.loads((ROOT / "samples" / "manifest.json").read_text(encoding="utf-8"))
            key = {t["q"]: t["expected"] for t in mf[0]["truth"]}
            self._send(200, json.dumps(key))
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True, "opencv": cv2.__version__}))
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path != "/grade":
            return self._send(404, "{}")
        ct = self.headers.get("Content-Type", "")
        if "multipart" not in ct:
            return self._send(400, json.dumps({"error": "multipart form with 'photo' + 'key'"}))
        boundary = ct.split("boundary=")[-1].encode()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        parts = body.split(b"--" + boundary)
        photo, key = None, {}
        for p in parts:
            if b"filename=" not in p and b'name="key"' not in p:
                continue
            header, _, payload = p.partition(b"\r\n\r\n")
            payload = payload.rsplit(b"\r\n", 1)[0]
            if b'name="key"' in header:
                try:
                    key = json.loads(payload.decode())
                except Exception:
                    key = {}
            elif b"filename=" in header:
                photo = np.frombuffer(payload, np.uint8)
        if photo is None:
            return self._send(400, json.dumps({"error": "no photo"}))
        bgr = cv2.imdecode(photo, cv2.IMREAD_COLOR)
        out = handle_photo(bgr, key)
        self._send(200, json.dumps(out))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"HomeworkHawk demo on http://127.0.0.1:{port} (OpenCV {cv2.__version__})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
