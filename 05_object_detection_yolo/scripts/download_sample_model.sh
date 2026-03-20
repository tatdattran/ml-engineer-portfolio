#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")
print("Downloaded sample model: yolov8n.pt")
PY
