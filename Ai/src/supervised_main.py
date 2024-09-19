import asyncio
import websockets
import json

async def handle_connection(websocket, path):
    try:
        async for message in websocket:
            print(f"Received message on 8003: {message}")
            data = json.loads(message)
            
            if data.get('code') == 'init':
                response = json.dumps({'code': 'supervised init'})
            elif data.get('code') == 'train':
                response = json.dumps({'code': 'resultat', 'data': 'result from 8003'})
            elif data.get('code') == 'stop':
                response = json.dumps({'code': 'stopped'})
            else:
                response = json.dumps({'code': 'unknown', 'message': 'Unknown code'})

            await websocket.send(response)
            print(f"Sent response on 8003: {response}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")

async def main():
    async with websockets.serve(handle_connection, "127.0.0.1", 8003):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
