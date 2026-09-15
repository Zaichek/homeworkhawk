/**
 * HomeworkHawk × Tavily — practice-problem finder.
 *
 * For every question worth reviewing (wrong or escalated), searches the web for
 * similar practice problems and returns per-question practice links. Feeds the
 * parent explanation ("here's what to practice tonight") and the CALL-E script.
 *
 * Track: Nebius x NVIDIA hackathon — Best Use of Tavily ($3,000).
 * Pipeline: OpenCV grade → Nemotron explain → Tavily practice → CALL-E deliver.
 *
 * Env: TAVILY_API_KEY (repo-root .env; free Researcher tier = 1,000 credits/mo)
 * Usage:
 *   node practice_finder.mjs ../call-e/graded.json --out practice.json
 *   node practice_finder.mjs ../call-e/graded.json --dry-run   # show queries, call nothing
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const args = process.argv.slice(2);
const file = args.find(a => !a.startsWith("--")) ?? "graded.json";
const flag = (n, d) => (args.indexOf(`--${n}`) >= 0 ? args[args.indexOf(`--${n}`) + 1] : d);
const DRY = args.includes("--dry-run");
const OUT = flag("out", null);

// repo-root .env (never committed)
const envPath = new URL("../.env", import.meta.url);
if (existsSync(envPath) && !process.env.TAVILY_API_KEY) {
  for (const line of readFileSync(envPath, "utf8").split("\n")) {
    const m = line.match(/^([A-Z_]+)=(.*)$/);
    if (m) process.env[m[1]] ??= m[2].trim();
  }
}
const KEY = process.env.TAVILY_API_KEY;

const run = JSON.parse(readFileSync(file, "utf8"));
const review = (run.questions ?? []).filter(q => ["wrong", "incorrect", "escalated", "unclear"].includes(q.verdict));

// query design: topic-level, not answer-level — we want drill material, not solutions
const queries = review.map(q => {
  const expected = String(q.expected ?? "");
  const digits = expected.replace(/\D/g, "").length;
  const topic = digits >= 4 ? "multi-digit arithmetic"
    : digits === 3 ? "three-digit addition subtraction"
    : digits === 2 ? "two-digit addition with carrying"
    : "grade 2 arithmetic practice";
  return {
    index: q.index,
    query: `grade 2 ${topic} practice problems worksheet`,
    escalated: q.verdict === "escalated" || q.verdict === "unclear",
  };
});

if (DRY || !KEY) {
  console.log(JSON.stringify({
    mode: KEY ? "dry-run" : "no-key",
    note: KEY ? "add --live behavior by removing --dry-run" : "set TAVILY_API_KEY (.env at repo root)",
    student: run.student_name ?? "the student",
    would_search: queries,
  }, null, 2));
  if (OUT) writeFileSync(OUT, JSON.stringify({ mode: "dry-run", would_search: queries }, null, 2));
  process.exit(0);
}

const out = { mode: "live", student: run.student_name ?? "the student", per_question: [] };
let credits_used = 0;

for (const q of queries) {
  const resp = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${KEY}` },
    body: JSON.stringify({
      query: q.query,
      max_results: 2,
      include_domains: ["k5learning.com", "superteacherworksheets.com", "iknowit.com", "math-drills.com", "education.com"],
      search_depth: "basic",
    }),
  });
  credits_used += 1;
  if (!resp.ok) {
    out.per_question.push({ index: q.index, query: q.query, error: `HTTP ${resp.status}` });
    continue;
  }
  const data = await resp.json();
  out.per_question.push({
    index: q.index,
    query: q.query,
    escalated: q.escalated,
    practice: (data.results ?? []).map(r => ({ title: r.title, url: r.url })),
  });
}

out.tavily_credits_used = credits_used;
console.log(JSON.stringify(out, null, 2));
if (OUT) writeFileSync(OUT, JSON.stringify(out, null, 2));
