"""
Netlify Serverless Function Handler for FoodieHelp AI.
Bridges Netlify AWS Lambda events to FastAPI using Mangum with path normalization.
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

_asgi_handler = Mangum(app, lifespan="off")


def handler(event, context):
    """
    Netlify Lambda Event Handler.
    Normalizes paths when requests are routed through Netlify redirects.
    """
    raw_path = event.get("rawPath") or event.get("path") or ""
    prefix = "/.netlify/functions/api"

    if raw_path.startswith(prefix):
        clean_path = raw_path[len(prefix):]
        if not clean_path.startswith("/"):
            clean_path = "/" + clean_path
        if not clean_path.startswith("/api") and clean_path != "/":
            clean_path = "/api" + clean_path

        event["path"] = clean_path
        event["rawPath"] = clean_path
        if "requestContext" in event and "http" in event["requestContext"]:
            event["requestContext"]["http"]["path"] = clean_path

    return _asgi_handler(event, context)
