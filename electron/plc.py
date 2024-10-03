from pyftdi.ftdi import Ftdi
from pyftdi.eeprom import FtdiEeprom
import pyftdi.serialext
import asyncio
import websockets
import json

class PCtoPLC:

    def __init__(self, url='ftdi:///1', baudrate=9600):
        self.ftdi = Ftdi()
        self.ftdi.open_from_url(url)
        self.eeprom = FtdiEeprom()
        self.eeprom.connect(self.ftdi)
        self.ftdi.set_cbus_direction(0b1111, 0b1111)
        self.port = pyftdi.serialext.serial_for_url(url, baudrate=baudrate)
        self.data = ''

    def write_to_plc(self, message):

        try:
            self.ftdi.set_cbus_gpio(0b1111)
            if type(message) is str:
                message = str.encode(message)
            elif type(message) is int:
                message = int.to_bytes(message)
            elif type(message) is bytes or type(message) is bytearray:
                pass
            else:
                raise Exception("invalid type, str, int or bytes accepted for write_to_plc")

            self.port.write(message)
        except self.port.serialutil.SerialException:

            self.read_from_plc()
            self.write_to_plc(message)

        return 1

    def read_from_plc(self, size=8):
        self.ftdi.set_cbus_gpio(0b0000)
        self.data = self.port.read(size)

async def send_receive(plc,websocket):
   
    while True:
        # Wait for data to be received
        received_data = await websocket.recv()

        # Deserialize the received data
        deserialized_data = json.loads(received_data)
        print('Received:', deserialized_data)

        if deserialized_data.code == "start":
            plc.write_to_plc("start")
        
        elif deserialized_data.code == "stop":
            plc.write_to_plc("stop")

        elif deserialized_data.code == "error":
            plc.write_to_plc("error")
        
        


async def python_plc(plc,websocket):

    while True:
        plc.read_from_plc()
        #envoie moi le bon code mais a titre d'exemple
        # Define a sample object to send
        data_to_send = {'code': 'start', 'data': {}}
        serialized_data = json.dumps(data_to_send)
        await websocket.send(serialized_data)


async def main():
 uri = "ws://127.0.0.1:8003"
 plc = PCtoPLC()
 async with websockets.connect(uri) as websocket:
    # Schedule both send_receive and print_loop to run concurrently
    pc_python_task = asyncio.create_task(send_receive(plc,websocket))
    python_plc = asyncio.create_task(plc,python_plc(websocket))

    # Wait for both tasks to run indefinitely
    await asyncio.gather(pc_python_task, python_plc)

# Run the main function
asyncio.run(main())


