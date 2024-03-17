import socket
from Model import YoloModel
from common.Constants import *

if __name__ == "__main__":

    # Adresse IP et port pour écouter les signaux
    HOST = '0.0.0.0'  # Écoute sur toutes les interfaces

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, SUPERVISED_PORT))
        s.listen()
        print(f"Attente du signal 'OK' sur le port {SUPERVISED_PORT}...")
        conn, addr = s.accept()
        with conn:
            print('Connecté par', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                if data.decode() == 'OK':
                    print("Signal 'OK' reçu, démarrage de l'entraînement...")
                    model = YoloModel()
                    model.generate_yaml()
                    model.train(epochs=EPOCHS, batch=BATCH, data=YAML_FILE)
