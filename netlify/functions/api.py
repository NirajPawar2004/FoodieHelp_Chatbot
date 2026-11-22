"""
Netlify Serverless Function Handler for FoodieHelp AI.
Bridges Netlify AWS Lambda events to FastAPI using Mangum.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is available in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mangum import Mangum
from app import app
from database.db import init_db
from database.seed import seed_database

# Ensure database is initialized in /tmp if running on Netlify serverless
try:
    init_db()
    seed_database(force_reseed=False)
except Exception as e:
    print(f"[Netlify] Startup DB initialization: {e}")

# Export AWS Lambda / Netlify Function handler
handler = Mangum(app, lifespan="off")
