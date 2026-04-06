import json
import logging
from typing import List, Dict, Any

from fastapi import WebSocket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebSocketManager")


class ConnectionManager:
    """
    Manages WebSocket connections for real-time push notifications.
    Supports broadcasting to all connected clients simultaneously.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and add it to the pool."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Active connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        #Remove a WebSocket connection from the pool.
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Active connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        #Send a JSON message to ALL connected clients. Automatically removes dead connections.
        dead_connections = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        # Clean up dead connections
        for dead in dead_connections:
            self.disconnect(dead)

        if self.active_connections:
            logger.info(
                f"Broadcast sent to {len(self.active_connections)} client(s): "
                f"{message.get('event_type', 'unknown')}"
            )


# Singleton instance used across the application
manager = ConnectionManager()