
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
        self.ftdi.set_cbus_direction(0b1111, 0b1111) #set le cbus en output
        self.port = pyftdi.serialext.serial_for_url(url, baudrate=baudrate)
        self.port.timeout = 0.05 #set timeout du read du bus rs485 
        self.data = b''
        self.isWriting = 0 

    def write_to_plc(self, message):
        
        
        try:   
            self.isWriting = 1 #change variable pour pas que la lecture et l'écriture se passe en meme temps
            self.ftdi.set_cbus_gpio(0b1111) #met output cbus a 1
            if type(message) is str:
                message = str.encode(message)
            elif type(message) is int:
                message = int.to_bytes(message)
            elif type(message) is bytes or type(message) is bytearray:
                pass
            else:
                raise Exception("invalid type, str, int or bytes accepted for write_to_plc")
            
            self.port.write(message) #ecrit le message
            self.isWriting = 0

        except:
            pass

        return 1

    def read_from_plc(self, size=8): #size = nb de bytes
            
            if self.isWriting == 0:
                self.ftdi.set_cbus_gpio(0b0000) # met output cbus a 0
                self.data = self.port.read(size)
                
            return self.data

        
async def read_write_PLC(plcToPC: PCtoPLC,websocket):

    allo = 0
    
    while True:
        
        data = b''
        data = plcToPC.read_from_plc(8)
        
        if (b'S' in data):
            data_to_send = {'code': 'error', 'data': {}}
            await websocket.send(data_to_send)

        if (b'R' in data):
            data_to_send = {'code': 'ready', 'data': {}}
            serialized_data = json.dumps(data_to_send)
            await websocket.send(serialized_data)

        if(data != b''):
            print(data, flush = True)
            allo += 1
            print(allo)
        await asyncio.sleep(0.5)    

       

        


async def read_Web(plcToPC,websocket):
    
    while True:     
                
            received_data = await websocket.recv() #read websocket et attend message
                # Deserialize the received data
            deserialized_data = json.loads(received_data) 
            print('Received:', deserialized_data, flush=True)

            #ecrire byte au plc selon message
            if deserialized_data['code'] == 'forward':
                plcToPC.write_to_plc(b'F') #17920 on plc
                deserialized_data['code'] == 'done'
            elif deserialized_data['code'] == "backward":
                plcToPC.write_to_plc(b'B') #16896 on plc
                deserialized_data['code'] == 'done'
        

async def main():
  
  uri = "ws://127.0.0.1:8003"
  pcToPlc = PCtoPLC()
  async with websockets.connect(uri) as websocket:
     # Schedule both send_receive and print_loop to run concurrently
     # Wait for both tasks to run indefinitely
     
     await asyncio.gather(read_write_PLC(pcToPlc,websocket),read_Web(pcToPlc,websocket))
    
# # Run the main function
asyncio.run(main())