"""
atlas/api/websocket_manager.py

WebSocket connection manager — the real-time bridge to the frontend.

How it works:
  1. Frontend connects: GET /ws/{task_id} → WebSocket
  2. ATLAS agents emit StreamEvent objects via broadcast()
  3. Manager serialises them to JSON and sends to all subscribers of that task_id

This is how the "Agent Feed" in the dashboard gets live updates without polling.

Phase 4 note: The frontend subscribes once and receives every event type:
  - agent.started / agent.completed / agent.failed
  - step.result (code being written)
  - test.result (pass/fail with output)
  - stream.token (character-by-character LLM streaming)
  - cost.update (running cost to the cent)
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from atlas.contracts import StreamEvent
from atlas.logging import get_logger

log = get_logger(__name__)


class ConnectionManager:
    """
    Manages active WebSocket connections, keyed by task_id.

    Multiple browser tabs can subscribe to the same task — all receive
    the same event stream. This is intentional: it lets you open the
    ATLAS dashboard on two monitors during a demo.
    """

    def __init__(self) -> None:
        # task_id → list of connected WebSockets
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket, task_id: str) -> None:
        """Accept a new WebSocket connection and register it."""
        await websocket.accept()
        self._connections[task_id].append(websocket)

        log.info(
            "ws_connected",
            task_id=task_id,
            total_connections=len(self._connections[task_id]),
        )

        # Immediately send a confirmation so the frontend knows it's live
        await websocket.send_json(
            {
                "event": "connection.established",
                "task_id": task_id,
                "message": "Connected to ATLAS event stream",
            }
        )

    def disconnect(self, websocket: WebSocket, task_id: str) -> None:
        """Remove a WebSocket from the registry (called on close/error)."""
        connections = self._connections.get(task_id, [])
        if websocket in connections:
            connections.remove(websocket)

        if not connections:
            # Clean up empty lists to prevent memory leak
            self._connections.pop(task_id, None)

        log.info(
            "ws_disconnected",
            task_id=task_id,
            remaining=len(self._connections.get(task_id, [])),
        )

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, event: StreamEvent) -> None:
        """
        Send a StreamEvent to all subscribers of event.task_id.

        Dead connections are silently removed — the frontend will reconnect.
        """
        connections = self._connections.get(event.task_id, [])
        if not connections:
            return

        payload = event.model_dump(mode="json")
        dead: list[WebSocket] = []

        for ws in connections:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(payload)
                else:
                    dead.append(ws)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws, event.task_id)

    async def broadcast_raw(self, task_id: str, data: dict) -> None:
        """Send a raw dict (for non-StreamEvent messages like ping/pong)."""
        connections = self._connections.get(task_id, [])
        for ws in connections:
            try:
                await ws.send_json(data)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def active_tasks(self) -> list[str]:
        return list(self._connections.keys())

    @property
    def total_connections(self) -> int:
        return sum(len(v) for v in self._connections.values())


# Singleton — imported everywhere that needs to broadcast events
ws_manager = ConnectionManager()
