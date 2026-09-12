"""HomeworkHawk for Alexa+ — simulated Alexa+ experience (web app).

Amazon Build, Ship, Shape track requirement: "New to MCP? Build a simulated
Alexa+ experience in a web app using your preferred agentic tool. ... your
repo still needs to include the simulation's source code, and your demo needs
to clearly show it working."

This is that simulation: an Echo-Show-style voice front-end that drives the
REAL HomeworkHawk grading pipeline. Voice in = Web Speech API recognition
(optional, buttons always work), grading = OpenCV 5 pipeline via /grade,
voice out = speechSynthesis reading the same script an Alexa+ skill would.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from src.app.web import handle_photo  # real pipeline entry
import cv2, numpy as np

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'samples')
MANIFEST = json.load(open(os.path.join(SAMPLES, 'manifest.json')))
SAMPLE = next(e for e in MANIFEST if e['file'] == 'w0_shadow.png')
ANSWER_KEY = {t['q']: t['expected'] for t in SAMPLE['truth']}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self._send(200, open(os.path.join(os.path.dirname(__file__), 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
        elif path == '/sample-list':
            self._send(200, json.dumps([{'file': e['file'], 'key': {t['q']: t['expected'] for t in e['truth']}} for e in MANIFEST[:6]]))
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/grade':
            # The "photo" the simulated Alexa+ device just captured. In the demo
            # the parent drops the worksheet under the Echo Show camera — here we
            # read the chosen sample the same way web.py reads an upload.
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n) or b'{}')
            fname = req.get('file') or SAMPLE['file']
            if fname not in {e['file'] for e in MANIFEST}:
                self._send(400, '{"error":"unknown sample"}'); return
            key = {t['q']: t['expected'] for t in next(e for e in MANIFEST if e['file'] == fname)['truth']}
            bgr = cv2.imread(os.path.join(SAMPLES, fname))
            result = handle_photo(bgr, key)
            self._send(200, json.dumps(result))
        else:
            self._send(404, '{"error":"not found"}')

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    print(f'HomeworkHawk for Alexa+ (simulated experience) → http://localhost:{port}')
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()
