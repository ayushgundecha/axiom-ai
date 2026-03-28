"""Tests for axiom/api/ — FastAPI REST API.

Tests all endpoints using FastAPI's TestClient.
Written TDD-style before implementation.
"""



class TestHealthEndpoint:
    """Health check must return status and metadata."""

    def test_health_returns_200(self) -> None:
        # Use TestClient for sync testing of async app
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "ok"
        assert "active_sessions" in data
        assert "registered_envs" in data


class TestEnvironmentsEndpoint:
    """List registered environments."""

    def test_list_envs(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/envs")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data["environments"], list)
        # At minimum, json env should be registered
        assert "json" in data["environments"]


class TestTasksEndpoint:
    """List available tasks."""

    def test_list_tasks(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/tasks")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data["tasks"], list)


class TestSessionEndpoints:
    """Session CRUD and step/observe/evaluate."""

    def test_create_session(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/sessions",
            json={
                "env_name": "json",
                "task_id": "create_and_complete",
            },
        )
        # May fail if task doesn't exist yet, but endpoint should exist
        assert resp.status_code in (200, 201, 404, 422)

    def test_step_nonexistent_session_returns_404(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/sessions/nonexistent/step",
            json={
                "type": "api_call",
                "value": "add_todo",
                "params": {"title": "Test"},
            },
        )
        assert resp.status_code == 404

    def test_observe_nonexistent_session_returns_404(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/sessions/nonexistent/observe")
        assert resp.status_code == 404

    def test_evaluate_nonexistent_session_returns_404(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post("/sessions/nonexistent/evaluate")
        assert resp.status_code == 404

    def test_delete_nonexistent_session_returns_404(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.delete("/sessions/nonexistent")
        assert resp.status_code == 404


class TestFullEpisodeViaAPI:
    """End-to-end test: create session -> step -> evaluate via API.

    This test requires the JSON environment and a task config to be available.
    It may need to be adjusted once the actual task configs are created.
    """

    def test_complete_episode(self) -> None:
        from starlette.testclient import TestClient

        from axiom.api.app import create_app

        app = create_app()
        client = TestClient(app)

        # Create session
        resp = client.post(
            "/sessions",
            json={
                "env_name": "json",
                "task_id": "create_and_complete",
            },
        )
        assert resp.status_code in (200, 201)
        session_id = resp.json()["session_id"]

        # Step: add a todo
        resp = client.post(
            f"/sessions/{session_id}/step",
            json={
                "type": "api_call",
                "value": "add_todo",
                "params": {"title": "Test todo"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data

        # Observe
        resp = client.get(f"/sessions/{session_id}/observe")
        assert resp.status_code == 200

        # Evaluate
        resp = client.post(f"/sessions/{session_id}/evaluate")
        assert resp.status_code == 200
        scores = resp.json()["scores"]
        assert "completion" in scores
        assert "efficiency" in scores
        assert "accuracy" in scores
        assert "safety" in scores

        # Get trajectory
        resp = client.get(f"/sessions/{session_id}/trajectory")
        assert resp.status_code == 200

        # Cleanup
        resp = client.delete(f"/sessions/{session_id}")
        assert resp.status_code in (200, 204)
