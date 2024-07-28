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
        self.heartbeat_interval = 10

    async def run(self):
        while True:
            try:
                async with websockets.connect(f"ws://{self.host}:{self.port}") as websocket:
                    self.websocket = websocket
                    await self.send_message({'code': 'init'})
                    await asyncio.gather(
                        self.receive_message(),
                        self.heartbeat()
                    )
            except (websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK,
                    websockets.exceptions.InvalidURI) as e:
                self.print(f"Connection error: {e}")
                self.print("Retrying connection...")
                await asyncio.sleep(5)  # Wait before retrying
            except Exception as e:
                self.print(f"Unexpected error: {e}")
                await asyncio.sleep(5)

    async def receive_message(self) -> None:
        while True:
            try:
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

            except websockets.exceptions.ConnectionClosedError as e:
                self.print(f"Connection closed by the server {e}")
                break
            except websockets.exceptions.ConnectionClosedOK:
                self.print("Connection closed normally.")
                break
            except Exception as e:
                self.print(f"An error occurred: {e}")
                break

    async def heartbeat(self):
        while True:
            try:
                await self.websocket.ping()
                await asyncio.sleep(self.heartbeat_interval)
            except websockets.exceptions.ConnectionClosed:
                self.print("Connection closed by server, restarting task")
                break

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
