import asyncio
import websockets
import json

async def send_receive():
    uri = "ws://127.0.0.1:8003"
    async with websockets.connect(uri) as websocket:
        # Define a sample object to send
        data_to_send = {'code': 'error', 'data': {'message': 'Bouton STOP appuyer'}}

        # Serialize and send the object
        # Serialize and send the object
        serialized_data = json.dumps(data_to_send)
        await websocket.send(serialized_data)

asyncio.get_event_loop().run_until_complete(send_receive())
