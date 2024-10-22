import asyncio
import websockets
import json

async def send_receive():
    uri = "ws://127.0.0.1:8002"
    async with websockets.connect(uri) as websocket:
        # Define a sample object to send
        data_to_send = {'code': 'init', 'data': ""}

        # Serialize and send the object
        serialized_data = json.dumps(data_to_send)
        await websocket.send(serialized_data)

        while True:
            # Wait for data to be received
            received_data = await websocket.recv()

            # Deserialize the received data
            deserialized_data = json.loads(received_data)
            print('Received:', deserialized_data)

            # Echo back the received data
            await websocket.send(json.dumps(deserialized_data))

asyncio.get_event_loop().run_until_complete(send_receive())
