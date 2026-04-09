import pytest
from fastapi.testclient import TestClient

# Import your FastAPI application
from src.api.server import app

# Create a virtual test client
client = TestClient(app)

def test_websocket_connection():
    """
    Test that the WebSocket endpoint accepts connections virtually
    without needing a live uvicorn server running on port 8000.
    """
    # The TestClient's websocket_connect bypasses the network layer
    with client.websocket_connect("/ws/alerts") as websocket:
        # If the connection is successful, the websocket object is returned.
        # If the route didn't exist or failed, it would raise an exception above.
        assert websocket is not None
        
        # Note: Since this is an automated CI test, we do not use a 'while True' loop.
        # We connect, assert it works, and cleanly exit the context manager.