"""End-to-end Strands demo: the agent grades a worksheet photo by orchestrating
OpenCV tools, evidence decides each next call. Run:
  set HH_GLM_KEY=... && python strands_agent/demo.py samples/w0_shadow.png
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE = sys.argv[1] if len(sys.argv) > 1 else 'samples/w0_shadow.png'
cfg = json.load(open(os.path.join(os.path.expanduser('~'), '.autoclaw', 'setting.json')))
KEY = os.environ.get('HH_GLM_KEY', cfg['apiKey'])
BASE = os.environ.get('HH_GLM_BASE', cfg['baseUrl'])

manifest = json.load(open('samples/manifest.json'))
entry = next(e for e in manifest if SAMPLE.endswith(e['file']))
answer_key = {t['q']: t['expected'] for t in entry['truth']}
truth = {t['q']: t['is_correct'] for t in entry['truth']}

from strands_agent.grader_agent import build_agent
agent = build_agent('openai/glm-5.3', KEY, BASE)

prompt = (f"Please grade the worksheet photo `{SAMPLE}`. "
          f"Answer key: {json.dumps({str(k): v for k, v in answer_key.items()})}. "
          f"Follow your policy step by step, then give the parent report.")
print(f'=== {entry["file"]} ===')
result = agent(prompt)
print('\n=== AGENT FINAL ANSWER ===')
print(str(result))
