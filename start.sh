#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  🫒  OLIVE AI ASSISTANT — Quick Start
# ═══════════════════════════════════════════════════════════

set -e

echo ""
echo "🫒  Olive AI Assistant — Hackathon Setup"
echo "════════════════════════════════════════"

# 1. Check Python
python3 --version || { echo "❌ Python 3 required"; exit 1; }

# 2. Install deps
echo ""
echo "📦  Installing dependencies..."
pip install -r requirements.txt -q

# 3. Set API key
if [ -z "$OPENAI_API_KEY" ]; then
  echo ""
  echo "⚠️  OPENAI_API_KEY not set!"
  echo "   LLM calls will fail until you set it:"
  echo "   export OPENAI_API_KEY='sk-proj-...'"
  echo "   Continuing for now (RAG and Diagnosis will work if models are local)..."
fi

# 4. Download corpus
echo ""
echo "📥  Downloading corpus..."
python3 download_corpus.py

# 5. Start backend (background)
echo ""
echo "🚀  Starting backend on http://localhost:8001 ..."
# Ensure we are in the right directory
mkdir -p frontend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload > server.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# 6. Wait for backend to be ready
echo "   Waiting for server to spin up..."
for i in {1..10}; do
  if curl -s http://localhost:8001/health > /dev/null; then
    echo "   ✅ Server is UP"
    break
  fi
  sleep 2
done

# 7. Index corpus (forced if index is missing)
echo ""
echo "🔍  Indexing corpus..."
curl -s -X POST http://localhost:8001/admin/index-corpus | python3 -m json.tool

# 8. Health check
echo ""
echo "❤️   Health check:"
curl -s http://localhost:8001/health | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "✅  Ready!  Open http://localhost:8001"
echo "   Server logs are in server.log"
echo "   Press Ctrl+C to stop"
echo "════════════════════════════════════════"

# Wait for backend process to finish
wait $BACKEND_PID
