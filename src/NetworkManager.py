import json
import socket
import asyncio
import threading
from  concurrent.futures import ThreadPoolExecutor

class NetworkManager():
    def __init__(self, worker: threading.Thread, host: str, port: str, verbose: bool = False):
        super().__init__()
        self.verbose = verbose

        self.host = host
        self.port = port

        self.worker = worker

        self.executor = ThreadPoolExecutor()
        self.loop = asyncio.get_event_loop()
        self.future = None

    def start(self):
        asyncio.ensure_future(self.run(), loop=self.loop)

        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
        finally:
            asyncio.get_event_loop().close()
    
    async def run(self):
        await self.socket_connect()
        await self.receive_message()
        await self.close_socket()

    async def receive_message(self) -> None:
        while True:
            data = await self.loop.sock_recv(self.s, 1024)
            received_json = json.loads(data.decode())
            code = received_json['code']['code']
            self.print(f'Received code: {code}')


            if self.future and self.future.done():
                self.print("NetworkManager : Finish Pipeline")
                await self.send_message(self.future.result())
                self.future = None

            if code == "start":
                self.print("NetworkManager : Start Pipeline")
                await self.send_message('start')
                self.future = self.loop.run_in_executor(self.executor, self.worker.start)

            elif code ==  "stop":
                self.print("NetworkManager : Stop Pipeline")
                self.worker.stop()
                await self.send_message('stop')

            else:
                await self.send_message('error')

    async def send_message(self, data) -> None:
        serialized_data = json.dumps(data).encode()
        await self.loop.sock_sendall(self.s, serialized_data)
        
    async def socket_connect(self) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setblocking(False)

        connected = False
        while not connected:
            try:
                await self.loop.sock_connect(self.s, (self.host, self.port))
                connected = True
                self.print("Successfully connected")
            except OSError as e:
                self.print(f"Error connecting: {e}")
                await asyncio.sleep(1)

    def close_socket(self) -> None:
        self.s.close()

    def print(self, str) -> None:
        if self.verbose:
            print(str)