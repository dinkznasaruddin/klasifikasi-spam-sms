#!/usr/bin/env bash
set -euo pipefail

# One-click evaluation runner with auto venv + requirements
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PY="${PROJECT_DIR}/.venv/bin/python"

ensure_venv_and_deps() {
  if [ ! -x "$PY" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv .venv
    PY="${PROJECT_DIR}/.venv/bin/python"
  fi
  echo "[setup] Ensuring dependencies from requirements.txt..."
  "$PY" -m pip install --upgrade pip >/dev/null
  if [ -f requirements.txt ]; then
    "$PY" -m pip install -r requirements.txt
  else
    "$PY" -m pip install numpy pandas scikit-learn matplotlib seaborn joblib requests
  fi
}

ensure_venv_and_deps

exec "$PY" src_evaluate.py
