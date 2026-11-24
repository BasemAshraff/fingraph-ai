#!/usr/bin/env bash
# Start FinGraph locally. Usage: ./run.sh   (then open http://localhost:8000)
set -e
cd "$(dirname "$0")/backend"
[ -f ../.env ] && set -a && . ../.env && set +a
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
