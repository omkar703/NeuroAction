#!/usr/bin/env bash
# ==============================================================================
# NeuroAction — Interactive Digital Twin & Research Showcase
# Unified Project Launcher (FastAPI Backend + Next.js Frontend)
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Ensure Node & uv/python are in PATH
export PATH="/home/op/.nvm/versions/node/v24.21.0/bin:$PATH"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Python virtual environment not found at $VENV_PYTHON"
    exit 1
fi

echo "=================================================================="
echo "NeuroAction — Neural Intention–Action Digital Twin"
echo "=================================================================="
echo "[*] Project Root: $PROJECT_ROOT"
echo "[*] Python:       $($VENV_PYTHON --version)"
echo "[*] Node.js:      $(node -v)"
echo "[*] npm:          $(npm -v)"
echo "------------------------------------------------------------------"

# Kill any existing processes on ports 8765 and 3000
echo "Checking ports 8765 and 3000..."
lsof -ti :8765 | xargs kill -9 2>/dev/null || true
lsof -ti :3000 | xargs kill -9 2>/dev/null || true

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down NeuroAction servers..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    echo "[OK] All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI Backend on port 8765
echo "Starting FastAPI backend service on http://127.0.0.1:8765..."
PYTHONPATH="$PROJECT_ROOT" "$VENV_PYTHON" -m uvicorn web.backend.main:app --host 127.0.0.1 --port 8765 &
BACKEND_PID=$!

# Wait for backend health check
echo "Waiting for backend to become ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8765/api/health > /dev/null 2>&1; then
        echo "[OK] FastAPI backend ready at http://127.0.0.1:8765 (API docs: http://127.0.0.1:8765/docs)"
        break
    fi
    sleep 0.5
done

# 2. Start Next.js Frontend on port 3000
echo "Starting Next.js frontend on http://localhost:3000..."
cd "$PROJECT_ROOT/web/frontend"
npm run start -- -p 3000 &
FRONTEND_PID=$!

echo ""
echo "=================================================================="
echo "NeuroAction Research Showcase is LIVE"
echo "=================================================================="
echo "  Web Application:   http://localhost:3000"
echo "  API Documentation: http://127.0.0.1:8765/docs"
echo "  API Health:        http://127.0.0.1:8765/api/health"
echo "------------------------------------------------------------------"
echo "  Interactive Pages:"
echo "    - Overview:         http://localhost:3000/"
echo "    - Science:          http://localhost:3000/science"
echo "    - Model Lab:        http://localhost:3000/model-lab"
echo "    - 3D Digital Twin:  http://localhost:3000/digital-twin"
echo "    - Explainability:   http://localhost:3000/explainability"
echo "    - Evidence:         http://localhost:3000/evidence"
echo "=================================================================="
echo "Press Ctrl+C to stop all services."

# Keep script running
wait
