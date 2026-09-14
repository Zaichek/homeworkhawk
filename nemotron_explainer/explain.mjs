/**
 * HomeworkHawk × Nemotron — the AI explanation layer for graded worksheets.
 *
 * Takes the grading pipeline's graded.json and asks a Nemotron model (served on
 * Nebius Token Factory, OpenAI-compatible API) to produce kid-friendly, hint-first
 * explanations for every question worth reviewing (wrong or escalated).
 *
 * Track: Nebius x NVIDIA Global AI Hackathon — Best Apps and Agents.
 *   vision leg  : HomeworkHawk (OpenCV 5, CPU, 0.24 s/page)
 *   reasoning   : Nemotron 3 on Token Factory (this file)
 *   delivery    : CALL-E phone callback (call-e/notify.mjs) or in-app
 *
 * Env: NEBIUS_API_KEY (Token Factory key) — only needed for --live
 *      NEBIUS_BASE_URL (default https://api.tokenfactory.nebius.com/v1)
 *      NEBIUS_MODEL (default nvidia/nemotron-3-nano — probe with probe_models.mjs)
 *
 * Usage:
 *   node explain.mjs <graded.json> --out explanations.json            # dry-run: renders prompts, calls nothing
 *   node explain.mjs <graded.json> --live --out explanations.json     # real Token Factory calls
 */
import { readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const file = args.find(a => !a.startsWith("--")) ?? "graded.json";
const flag = (n, d) => (args.indexOf(`--${n}`) >= 0 ? args[args.indexOf(`--${n}`) + 1] : d);
const LIVE = args.includes("--live");
const OUT = flag("out", null);
const BASE = process.env.NEBIUS_BASE_URL ?? "https://api.tokenfactory.nebius.com/v1";
const MODEL = process.env.NEBIUS_MODEL ?? "nvidia/nemotron-3-nano";

const run = JSON.parse(readFileSync(file, "utf8"));
const qs = run.questions ?? [];
const review = qs.filter(q => q.verdict === "wrong" || q.verdict === "incorrect" || q.verdict === "escalated" || q.verdict === "unclear");

// ---- prompt: advisory, hint-first, never just hands over the answer ----
const SYSTEM = [
  "You are a warm elementary-math tutor helping a parent review homework WITH their child (approx. age 8-9).",
  "For each question you receive: the exercise, the child's answer, the expected answer, and the grading system's confidence.",
  "Write in simple English a 9-year-old can follow.",
  "Rules you must never break:",
  "1) Hint-first: start with a question or nudge that helps the child find their own mistake. Do NOT reveal the final answer in the hint.",
  "2) After the hint, add a short 'why' section the PARENT can use to explain the concept (this may state the correct method).",
  "3) No judgment of the child; praise the attempt, target the concept, not the person.",
  "4) If confidence < 0.8 or the row is escalated, say the handwriting was hard to read and ask the child to walk through their steps aloud instead of assuming the answer.",
  "5) Keep each explanation under 90 words. Output strict JSON, no markdown fences.",
].join("\n");

function exerciseFor(q) {
  // Worksheet rows are two-operand arithmetic. The pipeline's result.json records
  // expected answers but not the child's read-back (roadmap: export printed row +
  // read candidates). State that honestly so the model never invents the answer.
  return q.printed ?? `(row ${q.index}) the grading system did not capture the child's exact marks; expected answer is "${q.expected}"`;
}

const userPayload = {
  student: run.student_name ?? "the student",
  questions: review.map(q => ({
    index: q.index,
    exercise: exerciseFor(q),
    child_answer: q.given ?? q.read ?? null,
    expected_answer: q.expected,
    confidence: q.score ?? null,
    escalated: q.verdict === "escalated" || q.verdict === "unclear",
  })),
};

function buildMessages() {
  return [
    { role: "system", content: SYSTEM },
    { role: "user", content: JSON.stringify(userPayload, null, 1) },
  ];
}

// ---------------- dry-run (default): show what would be asked ----------------
if (!LIVE) {
  const msgs = buildMessages();
  const out = {
    mode: "dry-run",
    model: MODEL,
    base_url: BASE,
    est_input_tokens: Math.ceil((msgs[0].content.length + msgs[1].content.length) / 4),
    would_review: userPayload.questions.length,
    messages: msgs,
  };
  console.log(JSON.stringify(out, null, 2));
  if (OUT) writeFileSync(OUT, JSON.stringify(out, null, 2));
  process.exit(0);
}

// ---------------- live: call Token Factory (OpenAI-compatible) ----------------
const key = process.env.NEBIUS_API_KEY;
if (!key) {
  console.error("Set NEBIUS_API_KEY (Nebius Token Factory) for --live. No call placed.");
  process.exit(1);
}

const resp = await fetch(`${BASE}/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
  body: JSON.stringify({
    model: MODEL,
    messages: buildMessages(),
    temperature: 0.3,
    max_tokens: 1600,
    response_format: { type: "json_object" },
  }),
});

if (!resp.ok) {
  console.error(`token factory error ${resp.status}:`, (await resp.text()).slice(0, 300));
  process.exit(1);
}

const data = await resp.json();
const raw = data.choices?.[0]?.message?.content ?? "{}";
let explanations;
try { explanations = JSON.parse(raw); } catch { explanations = { raw }; }

const result = {
  mode: "live",
  model: MODEL,
  usage: data.usage ?? null,
  student: userPayload.student,
  explanations,
};
if (OUT) writeFileSync(OUT, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
