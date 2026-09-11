# Devpost 提交材料包（HomeworkHawk · OpenCV AI Competition 2026）

> 状态：**报名已完成**（1476→1480号参赛者之一），**项目页创建被 reCAPTCHA 拦住**（自动化点击无反应，需要人工点一次）。
> 用户操作：登录 devpost.com（浏览器已有 Google 登录态）→ 打开
> https://devpost.com/submit-to/30984-opencv-ai-competition-2026-powered-by-aws/manage/submissions
> → 点「Create project」→ 把下面内容逐字段粘贴 → Submit。

---

## 项目名
HomeworkHawk — an agentic vision homework grader

## 一句话简介（Inspiration / What it does 开头用）
One photo of a finished math worksheet → a graded page in 0.24 s, on CPU, with human-in-the-loop escalation for anything ambiguous.

## 详细描述（What it does / How we built it，直接贴）

HomeworkHawk turns a single phone photo of a completed elementary math worksheet into a graded page: per-question ✓/✗, digit-level error localization ("the 3rd digit should be 8"), and a heat-map overlay — in 0.24 seconds, pure OpenCV 5 on CPU.

**Why it is agentic, not just a pipeline.** Every OpenCV 5 stage is a callable tool and a decision policy sits on top. Vision evidence changes the next action:
- blur / glare metrics fail the quality gate → the system asks the parent to re-shoot
- page quadrilateral not found → re-run detection with looser Canny parameters
- per-digit confidence < 0.72 → re-binarize the row (Otsu) and re-read — a second OpenCV call decided by what the first one saw
- still uncertain → escalate that question to the parent — the system says "not sure" instead of guessing about a child's work

Every decision lands in a replayable trace (tool, parameters, perception, decision) — see evidence/*/result.json in the repo.

**How we built it (OpenCV 5 core):**
1. Quality gate — Laplacian-variance blur, clipped-white glare, darkness ratios
2. Page rectification — Canny → contours → approxPolyDP quad → getPerspectiveTransform/warpPerspective (IoU 0.998)
3. Illumination flattening — background-estimate division (kills phone shadows)
4. Print/handwriting separation — two-level intensity classification (toner ≪ pencil) + dilated-print subtraction
5. Question segmentation — projection profiles, ruling-line exclusion, question-number header filter, answer zone right of the "="
6. Answer reading — connected-component glyphs, fragment-only merging, aspect-preserved fixed-grid cosine vs digit-template banks
7. Agentic loop — per-digit confidence gates re-binarize / re-read / escalate

**AWS:** the identical handler (shared byte-for-byte) runs on Lambda behind a Function URL with an opencv-python-headless 5 layer (Graviton or x86_64); S3 for per-stage evidence with lifecycle expiry; CloudWatch for per-stage metrics. Free-tier sized: 0.15–0.30 s CPU per page.

**Measured results** (18 labeled synthetic worksheets × 6 degradations — perspective warp, diagonal shadow, blur, faint pencil, half-erased answer — 180 questions, 30% wrong answers injected):

| Metric | Result |
|---|---|
| Page-detection IoU | 0.998 |
| Question segmentation | ≈100% clean/shadow/faint/erased; 81.1% overall |
| Handwritten answer exact-read | 93.3% |
| Grading decision accuracy | 85.6% |
| Blur photos correctly rejected | 3/3 |
| Latency | 0.15–0.30 s/page, CPU |

**Published failure cases:** perspective-warped edges lose 1–2 rows; the template reader is font-biased (unusual handwriting escalates rather than errors silently); half-erased pencil ghosts sometimes still read. All runs including failures are in evidence/.

**Responsible use:** transient processing, no faces required, evidence expires, human approval gates every ambiguous verdict, and the product grades + explains — it never solves the exercise for the child.

## Challenges we ran into
Separating pencil from toner under household shadows (solved by flatten-then-two-level-intensity); not clipping taller-than-print handwriting during row segmentation; and honest escalation design — making the system comfortable saying "not sure".

## Accomplishments
0.24 s/page end-to-end on CPU with zero ML-model dependencies, and an agentic loop where every OpenCV re-analysis pass is logged and replayable.

## What we learned
That a well-tuned classical OpenCV pipeline plus a confidence-gated action policy can deliver a genuinely useful product loop without any GPU — and that publishing failure cases is a feature.

## What's next for HomeworkHawk
Multi-line answers, fractions, drawn figures; a small on-device digit classifier to augment the template bank (traces already record the training crops); the COOL/Graviton benchmark path.

## Built with
Python · OpenCV 5 · NumPy · Pillow · AWS Lambda / S3 / CloudWatch

## 链接
- Repo: https://github.com/Zaichek/homeworkhawk
- Demo video (3:35): https://github.com/Zaichek/homeworkhawk/raw/master/docs/video/HomeworkHawk-demo.mp4
- Technical report: https://github.com/Zaichek/homeworkhawk/blob/master/docs/report.md
- Architecture: https://github.com/Zaichek/homeworkhawk/blob/master/docs/architecture.png

## 提交清单核对（Final Submission Requirements 对照）
- [x] 技术报告 → docs/report.md
- [x] 代码仓（公开）→ github.com/Zaichek/homeworkhawk
- [x] 依赖锁定+构建/运行/测试说明 → README + requirements 在 README（opencv5/numpy/pillow 三件套）
- [x] 架构图 → docs/architecture.png
- [x] 可用 web 端点 → 本地 demo（README 三行命令起服务）；AWS 部署包 aws/README.md（**Lambda 实际部署需用户 AWS 账号，一天内可补**；如需 judge 在线端点，建议提交前部署）
- [x] ≤5分钟视频（公开链接）→ 3:35，GitHub 直链
- [x] 评估证据含失败案例 → evidence/ 全量 + report §4
