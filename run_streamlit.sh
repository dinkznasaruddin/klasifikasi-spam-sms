#!/usr/bin/env bash
set -euo pipefail

# Simple runner for Streamlit app with auto-setup of virtualenv and deps
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
    # Fallback minimal deps
    "$PY" -m pip install streamlit joblib scikit-learn numpy pandas
  fi
}

ensure_venv_and_deps

# Forward any extra args to streamlit (e.g., --server.port=8501)
exec "$PY" -m streamlit run app_streamlit.py "$@"
