import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets
from common.Constants import *

class NetworkManager():
    def __init__(self, worker: threading.Thread, server_host: str, server_port: str, unsupervised_host: str, unsupervised_port: str, supervised_host: str, supervised_port: str, verbose: bool = False):
        self.verbose = verbose
        self.server_host = server_host
        self.server_port = server_port
        self.unsupervised_host = unsupervised_host
        self.unsupervised_port = unsupervised_port
        self.supervised_host = supervised_host
        self.supervised_port = supervised_port
        self.worker = worker
        self.executor = ThreadPoolExecutor()
        self.loop = asyncio.get_event_loop()
        self.future = None
        self.heartbeat_interval = 10
        self.websockets = {}

    async def run(self):
        tasks = [
            asyncio.create_task(self.connect_service('server')),
            asyncio.create_task(self.connect_service('supervised')),
            asyncio.create_task(self.connect_service('unsupervised'))
        ]
        await asyncio.gather(*tasks)

    async def connect_service(self, service_name):
        if service_name == 'server':
            host = self.server_host
            port = self.server_port
        elif service_name == 'supervised':
            host = self.supervised_host
            port = self.supervised_port
        elif service_name == 'unsupervised':
            host = self.unsupervised_host
            port = self.unsupervised_port
        
        while True:
            try:
                async with websockets.connect(f"ws://{host}:{port}") as websocket:
                    self.websockets[service_name] = websocket
                    await self.send_message(service_name, {'code': 'init'})
                    await asyncio.gather(
                        self.receive_message(service_name),
                        self.heartbeat(service_name)
                    )
            except (websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK,
                    websockets.exceptions.InvalidURI) as e:
                self.print(f"Connection error ({service_name}): {e}")
                self.print("Retrying connection...")
                await asyncio.sleep(5)
            except Exception as e:
                self.print(f"Unexpected error ({service_name}): {e}")
                await asyncio.sleep(5)

    async def receive_message(self, service_name) -> None:
        websocket = self.websockets.get(service_name)
        while True:
            try:
                data = await websocket.recv()
                received_json = json.loads(data)
                code = received_json['code']
                self.print(f'Received code ({service_name}): {code}')

                if self.future and self.future.done():
                    await self.send_message(service_name, self.future.result())
                    self.future = None

                if code == "start":
                    await self.send_message(service_name, {'code': 'start'})
                    self.future = self.loop.run_in_executor(self.executor, self.worker.start)
                    result = await self.future
                    await self.send_message(service_name, {'code': 'resultat', 'data': result})

                elif code == "stop":
                    self.worker.stop()
                    await self.send_message(service_name, {'code': 'stop'})
                
                elif code == "train":
                    await self.send_message(service_name, {'code': 'train'})

            except websockets.exceptions.ConnectionClosedError as e:
                self.print(f"Connection closed by the {service_name}: {e}")
                break
            except websockets.exceptions.ConnectionClosedOK:
                self.print(f"Connection closed normally ({service_name}).")
                break
            except Exception as e:
                self.print(f"An error occurred ({service_name}): {e}")
                break

    async def heartbeat(self, service_name):
        websocket = self.websockets.get(service_name)
        while True:
            try:
                await websocket.ping()
                await asyncio.sleep(self.heartbeat_interval)
            except websockets.exceptions.ConnectionClosed:
                self.print(f"Connection closed ({service_name}), stopping heartbeat.")
                break

    async def send_message(self, service_name, data) -> None:
        websocket = self.websockets.get(service_name)
        if websocket:
            serialized_data = json.dumps(data)
            await websocket.send(serialized_data)
            self.print(f"Sent message to {service_name}")

    def start(self):
        asyncio.ensure_future(self.run(), loop=self.loop)
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
        finally:
            self.loop.close()

    def print(self, message) -> None:
        if self.verbose:
            print(message)


