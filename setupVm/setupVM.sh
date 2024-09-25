#!/bin/bash


# 0. Installer les dépendances
sudo apt update -y
sudo apt upgrade -y


#Installer docker
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
sudo apt-get update -y
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo 'ALL ALL=(ALL) NOPASSWD: /usr/bin/dofa' | sudo tee -a /etc/sudoers

# 1. Installer Dofa
sudo apt install ../electron/dist/dofa_*_amd64.deb
# 1. Activer Docker au démarrage
sudo systemctl enable docker
sudo systemctl start docker
# 2. Vérifier si le conteneur Docker 'postgres_container' existe
if [ "$(sudo docker ps | grep postgres_container)" ]; then
    echo "Le conteneur 'postgres_container' existe."
else
    # Démarrer le conteneur Docker 'postgres_container et pgadmin4' avec Docker Compose à partir du chemin spécifié
    DOFA_COMPOSE_PATH=~/PMC/docker-compose.yml
    if [ -f $DOFA_COMPOSE_PATH ]; then
        sudo docker-compose -f $DOFA_COMPOSE_PATH up -d
        echo "Les conteneurs 'postgres_container et pgadmin4' a été démarré avec Docker Compose."
    else
        echo "Le fichier Docker Compose n'a pas été trouvé à '$DOFA_COMPOSE_PATH'. Veuillez spécifier le chemin correct."
        exit 1
    fi
fi

# 3. Copier le fichier dofa.service au bon emplacement
DOFA_SERVICE_PATH="/etc/xdg/autostart/dofa.desktop"
sudo cp dofa.desktop "$DOFA_SERVICE_PATH"


# 4. Ajouter l'application 'dofa' en tant que service pour le démarrage au démarrage de l'ordinateur
sudo chmod +rw /etc/xdg/autostart/dofa.desktop

# 5. Désactiver les notifications de mise à jour
gsettings set com.ubuntu.update-notifier no-show-notifications true
sudo apt-get remove update-notifier -y

#Desactive le password sudo pour l'application
echo 'ALL ALL=(ALL) NOPASSWD: /usr/bin/dofa' | sudo tee -a /etc/sudoers
# script terminé
echo "Le script a terminé l'exécution avec succès."


