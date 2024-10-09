
import asyncio
import websockets
import json
import time
from pyftdi.ftdi import Ftdi
from pyftdi.eeprom import FtdiEeprom
import pyftdi.serialext

writtingToPlc = 0
data_to_send  = {'code': 'init', 'data': {}}
data_to_receive = {'code': 'init', 'data': {}}
write = 0
class PCtoPLC:

    def __init__(self, url='ftdi:///1', baudrate=9600):
        self.ftdi = Ftdi()
        self.ftdi.open_from_url(url)
        self.eeprom = FtdiEeprom()
        self.eeprom.connect(self.ftdi)
        self.ftdi.set_cbus_direction(0b1111, 0b1111)
        self.port = pyftdi.serialext.serial_for_url(url, baudrate=baudrate)
        self.port.timeout = 0.5
        self.data = b''
        self.isWriting = 0

    def write_to_plc(self, message):
        
        self.isWriting = 1
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
            self.isWriting = 0

        except:
            pass

        return 1

    def read_from_plc(self, size=8):
            try:
                self.ftdi.set_cbus_gpio(0b0000)
                self.data = self.port.read(size)
            except:
                self.read_from_plc(size)
            return self.data

""" async def send_receive(plc,websocket):
    
    
    while True:
        # Wait for data to be received
        print('write',flush=True)
        plc.write_to_plc(b'1')
        
        print('allo')
        received_data = await websocket.recv()
        print('recv',flush=True)
        # Deserialize the received data
        deserialized_data = json.loads(received_data)
        print('Received:', deserialized_data)

        if deserialized_data.code == 'forward':
           
            plc.write_to_plc(b'F')

        elif deserialized_data.code == "backward":
            pass
            #when sent S, plc read 16896
            plc.write_to_plc(b'B') #stop """
        
async def read_write_PLC(plcToPC: PCtoPLC,websocket):

    
    print('caca',flush=True)
    dataToPLC = b'D'
    data = b'N'
    while True:
        
        data = plcToPC.read_from_plc(8)
        if data is None:
            data = b'N'
        
        if (b'S' in data):
            data_to_send = {'code': 'error', 'data': {}}
            await websocket.send(data_to_send)

        if (b'R' in data):

            data_to_send = {'code': 'ready', 'data': {}}
            serialized_data = json.dumps(data_to_send)
            await websocket.send(serialized_data)
        await asyncio.sleep(0.05)    

       

        


async def read_Web(plcToPC,websocket):
    write = 1
    data_to_send = {'code': 'ready', 'data': {}}
    serialized_data = json.dumps(data_to_send)
    write = 0
    await websocket.send(serialized_data)
    while True:
                
                
                received_data = await websocket.recv()
                # Deserialize the received data
                deserialized_data = json.loads(received_data)
                print('Received:', deserialized_data, flush=True)

                if deserialized_data['code'] == 'forward':
                    plcToPC.write_to_plc(b'F')
                    deserialized_data['code'] == 'done'
                elif deserialized_data['code'] == "backward":
                    plcToPC.write_to_plc(b'B')
                    deserialized_data['code'] == 'done'
           



""" async def python_plc(plc,websocket):

    
    data_to_send = {'code': 'ready', 'data': {}}
    serialized_data = json.dumps(data_to_send)
    
    await websocket.send(serialized_data)
    while True:

        if (plc.isWriting == 0):
            plc.read_from_plc()
            
            #envoie moi le bon code mais a titre d'exemple
            # Define a sample object to send
            if 1:
                
                data = {}
                if plc.data != b'':
                    print(plc.data)
                    if b'R' in plc.data: 
                        data = 'ready'
                    elif b'E' in plc.data:
                        data = 'error'
                    data_to_send = {'code': data, 'data': {}}
                    serialized_data = json.dumps(data_to_send) """
                    

            
                       
                        


async def main():
  
  uri = "ws://127.0.0.1:8080"
  pcToPlc = PCtoPLC()
  async with websockets.connect(uri) as websocket:
     # Schedule both send_receive and print_loop to run concurrently
     # Wait for both tasks to run indefinitely
     
     await asyncio.gather(read_write_PLC(pcToPlc,websocket),read_Web(pcToPlc,websocket))
    
# # Run the main function
asyncio.run(main())


