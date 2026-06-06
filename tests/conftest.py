"""
tests/conftest.py

Shared pytest fixtures.

Why fixtures?
  - DRY: every test that needs an HTTP client gets the same one
  - Isolation: each test gets a fresh TaskService (no shared state)
  - Speed: fixtures can be scoped (function/module/session)

Fixtures defined here are automatically available in all test files.
No imports needed in test files.
"""

import pytest
from fastapi.testclient import TestClient

from atlas.api.app import create_app


@pytest.fixture(scope="function")
def app():
    """
    Fresh FastAPI application per test.

    scope="function" means each test function gets its own app instance
    with an empty TaskService. Tests are fully isolated.
    """
    return create_app()


@pytest.fixture(scope="function")
def client(app) -> TestClient:
    """
    Synchronous HTTP test client.

    FastAPI's TestClient wraps httpx and handles the async event loop.
    It also runs the lifespan (startup/shutdown) correctly.

    Usage:
        def test_something(client):
            response = client.get("/health")
            assert response.status_code == 200
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def sample_task_request() -> dict:
    """A valid task request body for reuse across tests."""
    return {
        "title": "Write a Python fibonacci function",
        "description": (
            "Write a Python function that computes the nth Fibonacci number "
            "using dynamic programming. Include docstring and type hints."
        ),
        "language": "python",
        "max_retries": 3,
        "budget_usd": 0.50,
    }
