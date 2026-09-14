/**
 * Probe Nebius Token Factory for available models — run once NEBIUS_API_KEY works,
 * pick the Nemotron model id for explain.mjs (set NEBIUS_MODEL).
 *
 * Usage: NEBIUS_API_KEY=... node probe_models.mjs [--nemotron-only]
 */
const BASE = process.env.NEBIUS_BASE_URL ?? "https://api.tokenfactory.nebius.com/v1";
const key = process.env.NEBIUS_API_KEY;
if (!key) { console.error("Set NEBIUS_API_KEY first."); process.exit(1); }

const resp = await fetch(`${BASE}/models`, { headers: { Authorization: `Bearer ${key}` } });
if (!resp.ok) { console.error(`error ${resp.status}:`, (await resp.text()).slice(0, 200)); process.exit(1); }
const data = await resp.json();
let ids = (data.data ?? []).map(m => m.id).sort();
if (process.argv.includes("--nemotron-only")) ids = ids.filter(i => /nemotron/i.test(i));
console.log(ids.join("\n"));
console.error(`\n${ids.length} model(s)`);
