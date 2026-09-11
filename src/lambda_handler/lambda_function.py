"""AWS Lambda adapter — the cloud twin of the local demo endpoint.

Exposes `lambda_handler(event, context)`:
  event = {"image_b64": "<base64 PNG/JPG>", "answer_key": {"1": "603", ...}}
  returns base64 graded overlay PNG + per-question verdict JSON + trace.

Deployment (documented in ../aws/README.md):
  - Layer: opencv-python-headless 5.x + numpy, built for Amazon Linux 2023
    x86_64/Graviton (the COOL path can reuse the same layer on Arm).
  - Function: this file + the src/ tree zipped as source.
  - Front: Lambda Function URL (or API Gateway). The S3 evidence bucket and
    DynamoDB run history are optional wiring via the same output dict.
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent import grader  # noqa: E402
from src.app.web import handle_photo  # noqa: E402


def lambda_handler(event, context):
    try:
        img_bytes = base64.b64decode(event["image_b64"])
        key = {int(k): str(v) for k, v in (event.get("answer_key") or {}).items()}
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": f"bad input: {e}"})}
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"statusCode": 400, "body": json.dumps({"error": "not an image"})}
    out = handle_photo(bgr, key)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(out),
    }
