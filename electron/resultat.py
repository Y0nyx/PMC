import asyncio
import websockets
import json

async def send_receive():
    uri = "ws://127.0.0.1:8002"
    async with websockets.connect(uri) as websocket:
        # Define a sample object to send
        data_to_send = {'code': 'resultat', 'data': {'resultat': False, 'url': '/imagesSoudure/soudure.png','boundingbox':'/imagesSoudure/soudure.txt','erreurSoudure':'1'}}

        # Serialize and send the object
        serialized_data = json.dumps(data_to_send)
        await websocket.send(serialized_data)

asyncio.get_event_loop().run_until_complete(send_receive())
