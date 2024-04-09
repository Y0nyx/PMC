import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets
from common.Constants import *

class NetworkManager():
    def __init__(self, worker: threading.Thread, host: str, port: str, verbose: bool = False):
        self.verbose = verbose
        self.host = host
        self.port = port
        self.worker = worker
        self.executor = ThreadPoolExecutor()
        self.loop = asyncio.get_event_loop()
        self.future = None

    async def run(self):
        async with websockets.connect(f"ws://{self.host}:{self.port}") as websocket:
            self.websocket = websocket
            await self.send_message({'code':'init'})
            await self.receive_message()

    async def receive_message(self) -> None:
        while True:
            print("received message")
            data = await self.websocket.recv()
            print("after recv")
            received_json = json.loads(data)
            print(received_json)
            code = received_json['code']
            self.print(f'Received code: {code}')

            if self.future and self.future.done():
                self.print("NetworkManager : Finish Pipeline")
                await self.send_message(self.future.result())
                self.future = None

            if code == "start":
                self.print("NetworkManager : Start Pipeline")
                await self.send_message({'code': 'start'})
                self.future = self.loop.run_in_executor(self.executor, self.worker.start)
                result = await self.future
                await self.send_message({'code': 'resultat', 'data': result})
                
            elif code == "stop":
                self.print("NetworkManager : Stop Pipeline")
                self.worker.stop()
                await self.send_message({'code': 'stop'})

    async def send_message(self, data) -> None:
        serialized_data = json.dumps(data)
        await self.websocket.send(serialized_data)
        self.print("Sent message")
        
    async def socket_connect(self) -> None:
        pass

    def close_socket(self) -> None:
        pass

    def start(self):
        asyncio.ensure_future(self.run(), loop=self.loop)
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
        finally:
            self.loop.close()

    def print(self, str) -> None:
        if self.verbose:
            print(str)
