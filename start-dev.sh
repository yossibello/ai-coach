#!/bin/bash
# ── AI Coach Dev Starter ──────────────────────────────────────────────────────
# Starts backend services via Docker and the frontend dev server locally.
# Run once: ./start-dev.sh
# Leave the terminal open. Next.js HMR will pick up all code changes automatically.
# Only re-run this script if Docker services crash.

set -e

echo "🐳 Starting backend services (postgres, redis, backend, worker)..."
docker-compose up -d postgres redis backend worker

echo ""
echo "⏳ Waiting for backend to be ready..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend ready"
    break
  fi
  sleep 1
done

echo ""
echo "🚀 Starting Next.js dev server..."
echo "   → App: http://localhost:3000"
echo "   → Changes to any file are picked up automatically (HMR)"
echo "   → Do NOT close this terminal while developing"
echo ""

cd "$(dirname "$0")/frontend"
npm run dev
