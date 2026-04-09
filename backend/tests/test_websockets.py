import asyncio
import websockets

async def listen():
    uri = "ws://localhost:8000/ws/alerts"
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket. Waiting for notifications...")
        while True:
            msg = await ws.recv()
            print(f"\nEvent received:\n{msg}")

asyncio.run(listen())