import asyncio
import websockets

async def test_websocket_connection():
    uri = "ws://8.134.150.174:8000/humanchat"
    async with websockets.connect(uri) as websocket:
        while True:
            message = input("Enter a message to send (or 'exit' to quit): ")
            if message == "exit":
                break
            await websocket.send(message)
            response = await websocket.recv()
            print(f"Received response: {response}")

asyncio.get_event_loop().run_until_complete(test_websocket_connection())