"""Pytest config — make `backend/` importable as the project root."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure `import app...` works from anywhere
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Use an isolated, in-memory-ish DB for any test that touches one
os.environ.setdefault(
    "DATABASE_URL", "sqlite+aiosqlite:///./test_aicoach.db"
)
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")
os.environ.setdefault("STRAVA_CLIENT_ID", "0")
os.environ.setdefault("STRAVA_CLIENT_SECRET", "test")
