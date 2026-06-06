"""
tests/unit/test_health.py

Tests for GET /health — the liveness check that Railway probes.

Rule: every public endpoint must have tests.
The health check is the most critical — if this breaks, Railway marks
the deployment as failed and rolls back.
"""

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    """Railway health probe must get a 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape(client: TestClient) -> None:
    """Health response must have the expected fields."""
    data = client.get("/health").json()

    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data
    assert "services" in data
    assert isinstance(data["services"], dict)


def test_health_services_field(client: TestClient) -> None:
    """Services dict must include at minimum 'tasks' and 'websocket'."""
    services = client.get("/health").json()["services"]
    assert "tasks" in services
    assert "websocket" in services


def test_health_is_fast(client: TestClient) -> None:
    """Health check must respond within 100ms (no blocking I/O)."""
    import time

    start = time.monotonic()
    client.get("/health")
    elapsed_ms = (time.monotonic() - start) * 1000

    assert elapsed_ms < 100, f"Health check too slow: {elapsed_ms:.0f}ms"
