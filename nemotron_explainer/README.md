# nemotron_explainer — HomeworkHawk × Nemotron

The AI explanation layer for graded worksheets: HomeworkHawk (OpenCV 5) grades
the page in 0.24 s on CPU, then a Nemotron model on **Nebius Token Factory**
writes hint-first, kid-friendly explanations for every question worth
reviewing — so the callback to the parent teaches the concept instead of
just reading scores.

**Track:** Nebius x NVIDIA Global AI Hackathon — Best Apps and Agents.
**Responsible-use line unchanged:** the product grades + explains; it never
solves the exercise for the child (hints first, answers only in the parent's
"why" section).

## Files

- `explain.mjs` — the layer. Dry-run by default (renders prompts, calls nothing); `--live` calls Token Factory.
- `probe_models.mjs` — list Token Factory models, pick the Nemotron id.

## Setup (once Nebius Builder Program credits arrive)

```bash
export NEBIUS_API_KEY=...          # Token Factory key (dev.nebius.com)
node probe_models.mjs --nemotron-only
export NEBIUS_MODEL=<pick from above>   # default nvidia/nemotron-3-nano
```

## Run

```bash
node explain.mjs ../call-e/graded.json                      # dry-run: prompts + token estimate
node explain.mjs ../call-e/graded.json --live --out explanations.json
```

## Data contract

Input: the grading pipeline's `result.json`/`graded.json`
(`{file, student_name, questions: [{index, expected, score, verdict, attempts}]}`).
Roadmap: export the printed row text and the child's read-back candidates from
the pipeline so hints can target the exact mistake; `explain.mjs` already
accepts optional `printed` / `given` per question.

## Cost

With the Nano class model: ~450 input tokens per worksheet (≈ $0.0003) —
a full class of 30 worksheets costs about one US cent.
