# AWS deployment guide (HomeworkHawk)

The cloud endpoint runs the **same** `handle_photo` as the local demo — behavior-identical.

## 1. Layer: OpenCV 5 (headless) + numpy

```bash
mkdir layer/python && cd layer/python
pip install -t . opencv-python-headless==5.0.0.93 numpy
cd .. && zip -r ../opencv5-layer.zip python
aws lambda publish-layer-version --layer-name opencv5 \
  --description "OpenCV 5 headless + numpy" \
  --zip-file fileb://../opencv5-layer.zip \
  --compatible-runtimes python3.12 \
  --compatible-architectures arm64 x86_64
```

Graviton (arm64) keeps you on the cheapest tier and is the natural stepping stone to
the COOL path (Cloud-Optimized OpenCV Library on AWS Marketplace).

## 2. Function

```bash
# zip the handler + src tree
zip -r hawk.zip src/lambda_handler/lambda_function.py src/ -x "*__pycache__*"
aws lambda create-function --function-name homeworkhawk \
  --runtime python3.12 --architectures arm64 --memory-size 512 --timeout 30 \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://hawk.zip --layers <layer-version-arn> \
  --role <execution-role-arn>
aws lambda create-function-url-config --function-name homeworkhawk \
  --auth-type NONE   # for the judge demo; put API Gateway in front for prod
```

## 3. Contract

```json
POST {function-url}/  {"image_b64": "...", "answer_key": {"1": "1081", "2": "1083"}}
→ {"quality": {...}, "seconds": 0.23, "questions": [...], "images": {"graded": "data:image/png;base64,..."}}
```

Free-tier sized: the pipeline is 0.15–0.30 s CPU on one core; a 512 MB function
processes thousands of sheets inside the permanent free allowance.

## 4. Evidence & observability wiring

- S3 bucket `homeworkhawk-evidence`: write `images.page/printed/handwriting/graded`
  per run, lifecycle rule expire after 7 days (children's homework — transient by design).
- CloudWatch embedded metrics: `stage_latency_ms{stage=gate|detect|separate|segment|read}`,
  `agent_attempts`, `escalations`.

## 5. Security notes

Function URL with `AuthType NONE` is acceptable only for the judge demo window; for any
real deployment put API Gateway + a per-family token in front, keep the bucket
private, and log image *hashes* rather than images if local law requires.
