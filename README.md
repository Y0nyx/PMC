# Projet DOFA

Bienvenue dans le répertoire de code du projet **DOFA**. Ce dépôt contient tout le code nécessaire au fonctionnement de la machine, organisé en différents modules pour faciliter le développement et l'exécution.

---

## Structure du Projet

### 1. **AI**  
Ce module contient le code lié aux réseaux de détection de défauts.  

**Pour exécuter l'IA :**  
```bash
python Ai/src/main.py
```

---

### 2. **Electron**  
Ce module gère l'interface utilisateur via Electron.  

**Pour lancer l'application :**  
```bash
npm run electron
```
*Assurez-vous de lancer cette commande depuis le dossier `electron`.*

---

### 3. **SetupVM**  
Ce dossier contient des scripts Bash pour configurer une machine virtuelle de développement sous Windows.

---

### 4. **SQL**  
Scripts SQL pour initialiser la base de données.

---

### 5. **Docker**  
Les fichiers Docker permettent d’exécuter le projet dans des conteneurs. C'est la manière qu'il faut lancer le code pour utiliser la machine, car cela permet de lancer tous les composants en même temps dans leur propre environnement.

**Pour lancer les conteneurs :**  
```bash
docker compose up
```

---

## Instructions Générales

1. Clonez ce dépôt sur votre machine :
   ```bash
   git clone <URL_DU_DEPOT>
   cd Projet-DOFA
   ```

2. Suivez les étapes spécifiques à chaque module selon vos besoins :
   - Lancez l'IA si vous travaillez sur les détections.
   - Lancez l'application Electron pour tester ou développer l'interface utilisateur.
   - Configurez votre environnement avec les scripts `setupVM` si vous développez sous Windows.
   - Initialisez la base de données avec les scripts SQL avant d’utiliser Docker.

3. Si vous souhaitez exécuter l’ensemble du projet via Docker, assurez-vous que Docker et Docker Compose sont installés, puis utilisez la commande `docker compose up`.

