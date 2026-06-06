"""
tests/unit/test_tasks.py

Tests for the Task API — POST /tasks, GET /tasks, GET /tasks/{id}.

Pattern: Arrange → Act → Assert
  - Arrange: set up the test state
  - Act: call the endpoint
  - Assert: check the response

These tests have no LLM calls — they test the API + service layer only.
"""

from fastapi.testclient import TestClient


class TestCreateTask:
    """POST /tasks"""

    def test_create_returns_202(self, client: TestClient, sample_task_request: dict) -> None:
        """Accepted, not Created — processing is async."""
        response = client.post("/tasks", json=sample_task_request)
        assert response.status_code == 202

    def test_create_returns_task_id(self, client: TestClient, sample_task_request: dict) -> None:
        data = client.post("/tasks", json=sample_task_request).json()
        assert "task_id" in data
        assert len(data["task_id"]) == 36  # UUID format

    def test_create_initial_status_is_pending(
        self, client: TestClient, sample_task_request: dict
    ) -> None:
        data = client.post("/tasks", json=sample_task_request).json()
        assert data["status"] == "pending"

    def test_create_returns_created_at(self, client: TestClient, sample_task_request: dict) -> None:
        data = client.post("/tasks", json=sample_task_request).json()
        assert "created_at" in data

    def test_create_invalid_title_rejected(self, client: TestClient) -> None:
        """Empty title must be rejected with 422."""
        response = client.post(
            "/tasks",
            json={"title": "", "description": "x" * 20, "language": "python"},
        )
        assert response.status_code == 422

    def test_create_short_description_rejected(self, client: TestClient) -> None:
        """Description must be >= 10 chars."""
        response = client.post(
            "/tasks",
            json={"title": "Test task", "description": "short", "language": "python"},
        )
        assert response.status_code == 422

    def test_create_negative_budget_rejected(self, client: TestClient) -> None:
        """Negative budget must be rejected."""
        body = {
            "title": "Test",
            "description": "A valid description here",
            "language": "python",
            "budget_usd": -1.0,
        }
        response = client.post("/tasks", json=body)
        assert response.status_code == 422


class TestGetTask:
    """GET /tasks/{task_id}"""

    def test_get_existing_task(self, client: TestClient, sample_task_request: dict) -> None:
        task_id = client.post("/tasks", json=sample_task_request).json()["task_id"]
        response = client.get(f"/tasks/{task_id}")
        assert response.status_code == 200

    def test_get_task_contains_correct_title(
        self, client: TestClient, sample_task_request: dict
    ) -> None:
        task_id = client.post("/tasks", json=sample_task_request).json()["task_id"]
        data = client.get(f"/tasks/{task_id}").json()
        assert data["title"] == sample_task_request["title"]

    def test_get_nonexistent_task_returns_404(self, client: TestClient) -> None:
        response = client.get("/tasks/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    def test_get_task_has_empty_artifacts_initially(
        self, client: TestClient, sample_task_request: dict
    ) -> None:
        task_id = client.post("/tasks", json=sample_task_request).json()["task_id"]
        data = client.get(f"/tasks/{task_id}").json()
        assert data["artifacts"] == []
        assert data["subtasks"] == []


class TestListTasks:
    """GET /tasks"""

    def test_list_empty_initially(self, client: TestClient) -> None:
        response = client.get("/tasks")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_created_tasks(
        self, client: TestClient, sample_task_request: dict
    ) -> None:
        client.post("/tasks", json=sample_task_request)
        client.post("/tasks", json=sample_task_request)

        tasks = client.get("/tasks").json()
        assert len(tasks) == 2

    def test_list_newest_first(self, client: TestClient, sample_task_request: dict) -> None:
        """Tasks should be returned newest-first."""
        id1 = client.post("/tasks", json=sample_task_request).json()["task_id"]
        id2 = client.post("/tasks", json=sample_task_request).json()["task_id"]

        tasks = client.get("/tasks").json()
        ids = [t["task_id"] for t in tasks]

        # id2 was created last, should appear first
        assert ids.index(id2) < ids.index(id1)

    def test_list_limit_parameter(self, client: TestClient, sample_task_request: dict) -> None:
        for _ in range(5):
            client.post("/tasks", json=sample_task_request)

        tasks = client.get("/tasks?limit=3").json()
        assert len(tasks) == 3


class TestCancelTask:
    """DELETE /tasks/{task_id}"""

    def test_cancel_pending_task(self, client: TestClient, sample_task_request: dict) -> None:
        task_id = client.post("/tasks", json=sample_task_request).json()["task_id"]
        response = client.delete(f"/tasks/{task_id}")
        assert response.status_code == 204

    def test_cancel_nonexistent_task_returns_404(self, client: TestClient) -> None:
        response = client.delete("/tasks/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
