import json
import socket
import asyncio
import threading

class NetworkManager():
    def __init__(self, worker: threading.Thread, host: str, port: str, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.s = None

        self.host = host
        self.port = port

        self.worker = worker

    def start(self):
        self.loop.run_until_complete(self.run())
    
    async def run(self):
        await self.socket_connect()

        await self.send_message({'code': 'stop', 'data': 'object data'})

        await self.receive_message()

        await self.close_socket()

    async def receive_message(self) -> None:
        while True:
            data = await self.loop.sock_recv(self.s, 1024)
            received_json = json.loads(data.decode())
            code = received_json['code']['code']
            self.print(f'Received code: {code}')

            if code == "start":
                self.worker.start()
            elif code ==  "stop":
                self.worker.stop()

    async def send_message(self, data) -> None:
        serialized_data = json.dumps(data).encode()
        await self.loop.sock_sendall(self.s, serialized_data)
        self.print("Sent message")
    
    async def check_stop_signal(self) -> bool:
        await asyncio.sleep(0)
        if self.s.recv(1024, socket.MSG_PEEK) == b'':
            return False
        else:
            await self.receive_message()
            return True
        
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