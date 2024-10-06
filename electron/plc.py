from pyftdi.ftdi import Ftdi
from pyftdi.eeprom import FtdiEeprom
import pyftdi.serialext
import asyncio
import websockets
import json
import time

writtingToPlc = 0

class PCtoPLC:

    def __init__(self, url='ftdi:///1', baudrate=9600):
        self.ftdi = Ftdi()
        self.ftdi.open_from_url(url)
        self.eeprom = FtdiEeprom()
        self.eeprom.connect(self.ftdi)
        self.ftdi.set_cbus_direction(0b1111, 0b1111)
        self.port = pyftdi.serialext.serial_for_url(url, baudrate=baudrate)
        self.port.timeout = 0.05
        self.data = ''
        self.isWriting = 0

    def write_to_plc(self, message):

        try:    
            writtingToPlc = 1
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
            writtingToPlc = 0

        except self.port.serialutil.SerialException:
            writtingToPlc = 1
            time.sleep(0.055)
            self.write_to_plc(message)
            writtingToPlc = 0
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

        if deserialized_data.code == "forward":

            #when sent G, plc read 17920
            plc.write_to_plc(b'F') 

        elif deserialized_data.code == "backward":
            
            #when sent S, plc read 16896
            plc.write_to_plc(b'B') #stop
        

async def python_plc(plc,websocket):

    while True:

        if not writtingToPlc:
            plc.read_from_plc()
            #envoie moi le bon code mais a titre d'exemple
            # Define a sample object to send
            print(data)
            data = {}
            if plc.data != b'':
                print(plc.data)
                if b'R' in plc.data: 
                    data = 'ready'
                elif b'E' in plc.data:
                    data = 'error'
                data_to_send = {'code': 'start', 'data': data}
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

# # Run the main function
asyncio.run(main())


