# Nebius × NVIDIA 第七赛提交套件（Best Apps and Agents track）

> 状态：文案全稿已就绪。绑卡→API key→`--live` 实测通过后，按 [RECORD] 标记录制、[FILL] 标记填数，即可提交。
> 截止 10-30 10:00 AM PT；反馈奖 Most Valuable Feedback $100×10。

---

## 一、Devpost 提交文案（直接粘贴）

### Project name
HomeworkHawk — agentic grading on open infrastructure

### Elevator pitch
OpenCV grades the worksheet in 0.24 s on CPU; Nemotron 3 on Nebius Token Factory turns the score into a tutor-style explanation; CALL-E phones it to the parent. Vision → reasoning → voice, all open models.

### About the project（Project Story）

## Inspiration

HomeworkHawk started as a pure-vision experiment: can a CPU-only OpenCV 5 pipeline grade a photographed math worksheet in under a second? It could — 0.24 s per page, 93.3% exact-read. But a graded page is not a lesson. Parents get a score and still ask "so what do I do with it?" The missing layer was reasoning: turn verdicts into guidance that helps a parent teach, without solving the exercise for the child.

## What it does

Three open legs, one workflow:

1. **Vision (HomeworkHawk, OpenCV 5, CPU):** phone photo → quality gate → page rectification (IoU 0.998) → print/handwriting separation → per-question grading with a confidence-gated policy that re-binarizes, re-reads, or escalates to a human. 0.24 s/page, zero ML dependencies.
2. **Reasoning (this repo, `nemotron_explainer/`):** the structured grading result goes to a **Nemotron model served on Nebius Token Factory** (OpenAI-compatible endpoint), which writes hint-first explanations: a Socratic nudge the child can follow, plus a "why" section the parent can use — explicitly forbidden from handing over the final answer in the hint, and instructed to never judge the child.
3. **Delivery (CALL-E):** an optional outbound phone call reads the advisory summary to the parent and books a review session — with a default no-call preview mode, guardian allowlist, and an ambiguity gate that stops the call when the vision leg is unsure.

The grading stays advisory everywhere: hints first, no grades as judgment, escalated rows become "check by hand", never "wrong".

## How we built it — and where Token Factory mattered

- The Nemotron layer is a thin, honest client: `explain.mjs` renders a strict-JSON tutoring prompt from the pipeline's `result.json`, calls `https://api.tokenfactory.nebius.com/v1/chat/completions` with `response_format: json_object`, and stores the explanations next to the worksheet record. A `--dry-run` mode renders the exact prompts and token estimate without spending anything — our whole test matrix costs about **one US cent per classroom** on the Nano class model.
- **Token Factory gave us three things we did not want to self-host:** an OpenAI-compatible API (the same client code also runs against any OpenAI-shaped endpoint, which made local CI trivial), a model catalog where switching Nano → Super → Ultra is a one-line `NEBIUS_MODEL` change (the vision leg escalates *rows*, the LLM leg can escalate *difficulty* — hard multi-step problems get a bigger Nemotron), and per-model pricing that made the cost story in this submission auditable rather than hand-wavy.
- **Why Nemotron:** the tutoring prompt needs to follow restrictive rules (hint-first, no answer leakage, strict JSON) more than it needs world knowledge. Nemotron's instruction-following on small models let us keep the whole pipeline on open weights — matching the spirit of a vision pipeline that already ran with zero ML-model dependencies.

## Significant updates during the Submission Period

HomeworkHawk's vision leg existed before this hackathon. Built **new** during the Submission Period: the entire `nemotron_explainer/` reasoning layer (prompt engineering, dry-run/live modes, token budgeting), the CALL-E safety rewrite (preview-by-default, guardian allowlist, ambiguity gate, output masking, idempotency), and the end-to-end wiring photo → grade → explanation → phone. The vision leg itself was untouched except for exporting the structured result contract the new layers consume.

## Measured results (vision leg, unchanged)

| Metric | Result |
|---|---|
| Page-detection IoU | 0.998 |
| Handwritten answer exact-read | 93.3% |
| Latency | 0.15–0.30 s/page, CPU |
| Blur/garbage photos rejected | 3/3 |

Reasoning leg (Token Factory): ~472 input tokens per worksheet (dry-run measured); explanations returned as strict JSON; hint-first rule compliance shown live in the demo video.

## Challenges

Making the LLM *not be the answer key*: the system prompt forbids revealing the expected value in the hint section, which fights the model's helpfulness priors — the structured "hint / why" split and JSON schema enforcement on Token Factory were what held the line. Also: keeping the whole chain honest when the vision leg is unsure — the ambiguity gate now blocks the phone call entirely rather than letting a confident-sounding voice read unreliable numbers.

## What's next

Read-back export from the vision leg (the exact digits the child wrote) so hints target the specific mistake; a Super-class escalation path for multi-step word problems; on-device Nano for privacy-sensitive households.

## Built with

Python · OpenCV 5 · NumPy · **NVIDIA Nemotron (open models) on Nebius Token Factory** · Node.js · CALL-E · AWS Lambda/S3

### Try it out links
- Repo: https://github.com/Zaichek/homeworkhawk （含 nemotron_explainer/）
- Video: [FILL: 录制后上传 YouTube，贴链接]

### Feedback（Most Valuable Feedback 奖素材，提交表单用）
- Token Factory 的 OpenAI 兼容端点让本地 CI 可以直接跑（无需改 SDK）；catalog 一行换模型的设计对成本分级很友好
- 建议改进：`response_format: json_object` 在小模型上偶发返回带 fence 的内容，希望加 schema 强校验开关；`/models` 列表建议标注 context+价格，方便脚本自动选型
- [FILL: live 实测后补 1-2 条真实体验]

---

## 二、3 分钟视频分镜脚本（[RECORD] = 需录制）

| 时间 | 画面 [RECORD] | 口播要点 |
|---|---|---|
| 0:00-0:20 | 手机拍作业照→成绩页 | "A photo of homework becomes a graded page in a quarter second — on a CPU. But a score isn't a lesson." |
| 0:20-0:50 | 终端跑 pipeline，graded.json 滚动 | Vision leg：OpenCV 5、质量门、透视校正、置信度门控升级人工；93.3% 读出率 |
| 0:50-1:40 | `node explain.mjs graded.json --live` 实况，返回 JSON 解释 | **Token Factory 时刻**：Nemotron 写 hint-first 讲解；展示"提示不给答案、why 段给家长"的规矩；472 token/卷≈零点几分钱 |
| 1:40-2:20 | 手机响铃（可用 call-e 实录素材风格）| CALL-E 电话播报 advisory 摘要+订复习时段；强调默认不打电话、白名单、歧义拦截 |
| 2:20-2:50 | Token Factory 模型目录页 | 开放基建叙事：Nano 管日常、一行换 Super/Ultra；全程开源模型，无自建推理 |
| 2:50-3:00 | GitHub 页收尾 | "Vision, reasoning, voice — three open legs. Repo in the description." |

录制清单：① 终端三连录屏（pipeline / explain --live / 目录页）② 手机拍照素材 ③ 电话环节可复用 call-e 实测风格（新录，别复用旧赛成片）。剪辑后传 YouTube（public、≤3:00），链接填上文 [FILL]。

---

## 三、绑卡后执行清单（agent 自动跑）
1. `NEBIUS_API_KEY=... node probe_models.mjs --nemotron-only` → 选 `NEBIUS_MODEL`（优先 nano）
2. `node explain.mjs ../call-e/graded.json --live --out explanations.json` → 人工检查 hint-first 合规
3. 按上表录制+剪辑+传 YouTube
4. Devpost manage/submissions 建项目（reCAPTCHA：先赌音频按钮，再走判格）
5. 反馈奖表单：填 Feedback 三条（上面已备好 2 条+live 后补 1 条）
6. 台账+记忆收尾
