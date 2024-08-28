import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets

from common.Constants import *

class NetworkManager():
    """
    This class manages the network connections with the other docker containers used in the system.
    ::
    Attributes:
        worker (threading.Thread): Object passed to the thread, that the thread will use to perform actions. The object needs to have the start method.
        verbose (bool): Flag for verbose prints.
    Methods:
        start: Starts all services.
        _connect_service: Connects to the specified service
        _receive_message: Listens and parses received messages from other services
        hearbeat
        
        
    """
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
        self._heartbeat_interval = 10
        self.websockets = {}

    async def _run(self) -> None:
        """
        Creates all necessary tasks to connect to the services.
        ::
        Args:
        Returns:
            None
        """
        tasks = [
            asyncio.create_task(self._connect_service('server')),
            asyncio.create_task(self._connect_service('supervised')),
            asyncio.create_task(self._connect_service('unsupervised'))
        ]
        await asyncio.gather(*tasks)

    async def _connect_service(self, service_name: str) -> None:
        """
        Connects to the specified service, sends the init message to the service and waits for a response while checking for a heartbeat.
        ::
        Args:
            service_name (str): Name of the service to connect.
        Returns:
            None
        """
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
                    await self._send_message(service_name, {'code': 'init'})
                    await asyncio.gather(
                        self._receive_message(service_name),
                        self._heartbeat(service_name)
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

    async def _receive_message(self, service_name: str) -> None:
        """
        Waits for a message, parses it when received and responds according to the specified code.
        ::
        Args:
            service_name (str): Name of the service to connect.
        Returns:
            None
        """
        websocket = self.websockets.get(service_name)
        while True:
            try:
                received = await websocket.recv()
                received_json = json.loads(received)
                code = received_json['code']
                self.print(f'Received code ({service_name}): {code}')

                if self.future and self.future.done():
                    await self._send_message(service_name, self.future.result())
                    self.future = None

                if code == "start":
                    await self._send_message(service_name, {'code': 'start'})
                    self.future = self.loop._run_in_executor(self.executor, self.worker.start)
                    result = await self.future
                    await self._send_message(service_name, {'code': 'resultat', 'data': result})

                elif code == "stop":
                    self.worker.stop()
                    await self._send_message(service_name, {'code': 'stop'})
                
                elif code == "train" and service_name == 'server':
                    try:
                        await asyncio.gather(self._heartbeat('supervised'))
                    except (websockets.exceptions.ConnectionClosedError,
                            websockets.exceptions.ConnectionClosedOK,
                            websockets.exceptions.InvalidURI,
                            AttributeError):
                        await self._send_message('server', {'code': 'error', 'data': 'supervised container disconnected'})

                    try:
                        await asyncio.gather(self._heartbeat('unsupervised'))
                    except (websockets.exceptions.ConnectionClosedError,
                            websockets.exceptions.ConnectionClosedOK,
                            websockets.exceptions.InvalidURI,
                            AttributeError):
                        await self._send_message('server', {'code': 'error', 'data': 'unsupervised container disconnected'})

                    await self._send_message('supervised', {'code': 'train'})
                    await self._send_message('unsupervised', {'code': 'train'})

                
                elif code == "resultat":
                    result = received_json['result']
                    await self._send_message('server', {'code': 'resultat', 'data': result})


            except websockets.exceptions.ConnectionClosedError as e:
                self.print(f"Connection closed by the {service_name}: {e}")
                break
            except websockets.exceptions.ConnectionClosedOK:
                self.print(f"Connection closed normally ({service_name}).")
                break
            except Exception as e:
                self.print(f"An error occurred ({service_name}): {e}")
                break

    async def _heartbeat(self, service_name: str) -> None:
        """
        Pings the specified service at a set interval and raises and exception if the service doesn't respond.
        ::
        Args:
            service_name (str): Name of the service to ping.
        Returns:
            None
        """
        websocket = self.websockets.get(service_name)
        while True:
            try:
                await websocket.ping()
                await asyncio.sleep(self._heartbeat_interval)
            except websockets.exceptions.ConnectionClosed:
                self.print(f"Connection closed ({service_name}), stopping _heartbeat.")
                break

    async def _send_message(self, service_name: str, data: dict) -> None:
        """
        Sends a message to the specified service.
        ::
        Args:
            service_name (str): Name of the service to send the message to.
            data (dict): Data to send in a dict following the format: {'code': code, 'data': data}
        Returns:
            None
        """
        websocket = self.websockets.get(service_name)
        if websocket:
            serialized_data = json.dumps(data)
            await websocket.send(serialized_data)
            self.print(f"Sent message {data} to {service_name}")

    def start(self) -> None:
        """
        Starts the network manager and all its tasks.
        ::
        Args:
        Returns:
            None
        """
        asyncio.ensure_future(self._run(), loop=self.loop)
        try:
            self.loop._run_forever()
        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
        finally:
            self.loop.close()

    def print(self, message) -> None:
        if self.verbose:
            print(message)


