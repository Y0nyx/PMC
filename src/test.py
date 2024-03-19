import socket

HOST = 'host.docker.internal'  # L'adresse IP du serveur
PORT = 8002         # Le port sur lequel le serveur écoute

# Créer un socket TCP/IP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Établir une connexion au serveur
    s.connect((HOST, PORT))
    print(f"Connexion établie avec succès au {HOST}:{PORT}")
except ConnectionRefusedError:
    print(f"Impossible de se connecter au {HOST}:{PORT}. Le serveur n'est peut-être pas démarré ou n'écoute pas sur ce port.")
finally:
    # Fermer la connexion
    s.close()

import socket

def scan_ports(target, start_port, end_port):
    open_ports = []
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)  # Timeout for the connection attempt
                result = s.connect_ex((target, port))
                if result == 0:
                    open_ports.append(port)
                    print(f"Port {port} is open")
        except KeyboardInterrupt:
            print("\nScan stopped by user.")
            break
        except socket.error:
            print("Couldn't connect to server.")
            break
    return open_ports

if __name__ == "__main__":
    target_host = '127.0.0.1' 
    start_port = 0
    end_port = 10000
    print(f"Scanning ports {start_port} to {end_port} on {target_host}...")
    open_ports = scan_ports(target_host, start_port, end_port)
    if open_ports:
        print("Open ports:", open_ports)
    else:
        print("No open ports found.")
