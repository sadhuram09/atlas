"""
main.py — ATLAS server entrypoint.

Run locally:     python main.py
Run with uvicorn: uvicorn main:app --reload --port 8000
Docker:          CMD ["python", "main.py"]
Railway:          Start command: python main.py
"""

import uvicorn

from atlas.api.app import app  # noqa: F401 — imported for uvicorn string ref
from atlas.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,  # Hot-reload in dev, off in prod
        log_level=settings.log_level.lower(),
        # Use uvicorn's built-in access log — structlog handles app logs
        access_log=True,
    )
