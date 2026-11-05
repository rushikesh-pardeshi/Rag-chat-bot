"""main.py

Role: Thin process entrypoint. Create the FastAPI app and run the server — no business logic.
"""

from __future__ import annotations

from backend.api.app import create_app

app = create_app()


def run() -> None:
    """Development entrypoint: `python main.py`. Production uses a real ASGI server."""
    import uvicorn

    from backend.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_config=None,  # logging is configured in backend/core/logging.py
    )


if __name__ == "__main__":
    run()
