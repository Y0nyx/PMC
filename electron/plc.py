
import asyncio
import websockets
import json
import time
from pyftdi.ftdi import Ftdi
from pyftdi.eeprom import FtdiEeprom
import pyftdi.serialext


class PCtoPLC:

    def __init__(self, url ='ftdi:///1', baudrate=9600):
        '''
        url : de l'adaptateur RS-485
        baudrate : vitesse d'envoie et de lecture
        '''
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
        '''
        Description : méthode pour écrire au PLC
        message : message à écrire
        '''
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
            #seul exception possible est quand on ecrit et lit en meme temps, c'est pas quelque chose
            #de possible avec la facon dont les trucs sont fait
            pass

        return 1

    def read_from_plc(self, size=8):
            '''
            Description : méthode pour lire du PLC
            size : nombre de byte a lire
            '''
            if self.isWriting == 0:

                self.ftdi.set_cbus_gpio(0b0000) # met output cbus a 0
                self.data = self.port.read(size)

            return self.data


async def read_write_PLC(plcToPC: PCtoPLC,websocket):
    '''
    Description : fonction pour lire du PLC et ecrire sur le websocket
    plcToPC : Classe pour la communication avec l'adaptateur RS-485
    websocket : connection au websocket
    '''


    time.sleep(1)

    init = 0
    while True:

        data = b''
        data = plcToPC.read_from_plc(8)
        
        if(data != b''):
            print(data, flush = True)


        
        if (b'S' in data):
            data_to_send = {'code': 'error', 'data': {}}
            serialized_data = json.dumps(data_to_send)

            await websocket.send(serialized_data)


        if (b'A' in data):

            print('A') #A qaund c'est arrivé au bout
            data_to_send = {'code': 'ready', 'data': {}}
            serialized_data = json.dumps(data_to_send)

            await websocket.send(serialized_data)

        if (b'E' in data):

            print('E') #E quand c'est revenu  vers le user
            data_to_send = {'code': 'ready', 'data': {}}
            serialized_data = json.dumps(data_to_send)

            init = 0
            await websocket.send(serialized_data)


        if (b'I' in data and init == 0):

            print('I', flush =True) # I pour init, quand c'est revenu et que le UI est au menu principal
           
            data_to_send = {'code': 'ready', 'data': {}}
            serialized_data = json.dumps(data_to_send)

            init = 1
            await websocket.send(serialized_data)
        
        if (b'T' in data and init == 0):

            print('T', flush =True) # I pour init, quand c'est revenu et que le UI est au menu principal
           
            data_to_send = {'code': 'ready', 'data': {}}
            serialized_data = json.dumps(data_to_send)

            await websocket.send(serialized_data)


        await asyncio.sleep(0.2)



async def read_Web(plcToPC : PCtoPLC ,websocket):
    '''
    Description : fonction pour lire du websocket et envoie au PLC une commande selon ce qui est lu
    plcToPC : Classe pour la communication avec l'adaptateur RS-485
    websocket : connection au websocket
    '''
    while True:
           
            received_data = await websocket.recv() #read websocket et attend message
            # Deserialize the received data
            
            deserialized_data = json.loads(received_data)

            print('Received:', deserialized_data, flush=True)

            #ecrire byte au plc selon message
            if deserialized_data['code'] == 'forward':

                print('forward', flush = True)
                plcToPC.write_to_plc(b'F') #17920 on plc
                deserialized_data['code'] == 'done'

            if deserialized_data['code'] == "backward":

                print('backward', flush = True)
                plcToPC.write_to_plc(b'B') #16896 on plc
                deserialized_data['code'] == 'done'

            if deserialized_data['code'] == "ready":

                print('ready', flush = True)
                plcToPC.write_to_plc(b'R') #????? on plc
                print('readySEND', flush = True)
                deserialized_data['code'] == 'done'

async def main():

  uri = "ws://127.0.0.1:8003"

  while True:

    pcWrite = PCtoPLC(url ='ftdi://0x403:0x6001/1' ) #classe pour write avec adaptateur #6001
    pcRead = PCtoPLC(url ='ftdi://0x403:0x6015/1' ) #classe pour write avec adaptateur #6015
    async with websockets.connect(uri) as websocket:
     # Schedule both send_receive and print_loop to run concurrently
     # Wait for both tasks to run indefinitely
      await asyncio.gather(read_write_PLC(pcRead,websocket),read_Web(pcWrite,websocket))

# # Run the main function
asyncio.run(main())
