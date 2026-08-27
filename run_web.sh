#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "web/frontend/dist" ]; then
  echo "building frontend..."
  (cd web/frontend && npm install && npm run build)
fi

echo "starting server on http://localhost:5050"
.venv/bin/python web/backend/app.py
