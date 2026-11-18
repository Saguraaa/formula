#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export FORMULA_DEPLOYMENT=cloud
export FORMULA_DEVICE=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PORT="${PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x .cloud-venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .cloud-venv
fi
PY=.cloud-venv/bin/python
fingerprint="$(cat requirements-cloud.txt deploy/start-cloud.sh | sha256sum | cut -d ' ' -f 1)"
if [[ ! -f .cloud-venv/.requirements-sha ]] || [[ "$(cat .cloud-venv/.requirements-sha)" != "$fingerprint" ]]; then
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip uninstall -y opencv-python-headless opencv-python || true
  "$PY" -m pip install -r requirements-cloud.txt
  printf '%s' "$fingerprint" > .cloud-venv/.requirements-sha
fi
"$PY" deploy/check_models.py
exec "$PY" -m gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 2 --timeout 300 --access-logfile - 'wsgi:app'
