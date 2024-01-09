# PMC
## Commencer à coder 

### Installation de wsl
Suivre les instructions d'installation de wsl sous la version Ubuntu-22.04, ne pas oublier d'upgrader à [wsl 2](https://learn.microsoft.com/fr-fr/windows/wsl/install).

Pour vérifier si votre installation s'est bien déroulée, exécutez la commande suivante dans le terminal :

```bash
wsl -l -v
```
Vous devriez obtenir la sortie suivante :
```bash
    NAME                   STATE           VERSION
  * Ubuntu-22.04           Running         2
```

### Installation Conda

Dans WSL, commencez par exécuter la commande suivante pour récupérer la dernière version de Miniconda :

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Exécutez ensuite cette commande en modifiant le chemin avec votre emplacement pour initialiser Conda :

```bash
./home/YOUR_NAME/miniconda3/bin/conda init
```

Fermez ensuite votre terminal, puis ouvrez-en un nouveau et redémarrez votre WSL :
```bash
wsl --shutdown
```

Ouvrez un nouveau WSL, votre ligne de commande devrait maintenant afficher le préfixe (base). Enfin, créez votre environnement de développement :

```bash
conda create -n PMC python=3.10.9
conda activate PMC
```

Accédez au projet PMC et exécutez la commande suivante :

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
```

** Essayez de minimiser l'ajout de bibliothèques pour rendre votre code aussi léger que possible.

Pour mettre à jour le fichier requirements.txt, utilisez la bibliothèque pipreqs avec la commande suivante. Cela analysera entièrement le projet et ajoutera les bibliothèques nécessaires.

`pipreqs /home/project/location`

## Flux de Travail (Workflow)

Durant le développement du PMC, nous allons utiliser la méthodologie GitFlow, qui est décrite en détail dans ce tutoriel: [GitFlow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

La branche principale "Main" doit être fusionnée lors de changements majeurs. C'est à partir de la branche "Dev" que nous développerons les nouvelles fonctionnalités de notre projet.

![GitFlow Diagram](https://github.com/Y0nyx/PMC/assets/72567319/91c8cedc-eb23-4f64-a4c2-6b9b6952b5b0)

## Standard de Code (PEP 8)

Pour assurer la qualité et la lisibilité de notre code, nous adhérons au standard PEP 8 pour le style du code Python. Vous pouvez consulter les détails de ce standard ici: [PEP 8 - Style Guide for Python Code](https://pep8.org/).

Voici quelques bonnes pratiques à respecter pour assurer une pull request de haute qualité avec un minimum de commentaires:

### Nombres Magiques

Evitez d'utiliser des indices non justifiés, car ils nuisent à la lisibilité du code et peuvent le rendre confus. Déclarez plutôt vos indices de cette façon : `path[4]` -> `path[name_index]`.

### Grand Calcul

Les conversions non commentées et les calculs non justifiés peuvent également rendre le code moins clair. Pour préserver la logique et améliorer votre code, veuillez créer des classes de conversion.

### Indentation

Veillez à éviter plus de trois niveaux d'indentation. Dans ce cas, il est possible qu'une fonction supplémentaire soit nécessaire.

## Structure du Projet

Respectez la structure du code établie préalablement lors de la création de ce dépôt GitHub. Elle devrait ressembler à quelque chose comme cela :

```
src/
├── tools/
│   ├── conversion/
│   │   └── ndarray2tensor.py
├── pipeline/
│   ├── camera/
│   │   ├── cameraManager.py
│   │   ├── cameraSensor.py
│   │   ├── laptopCamera.py
│   │   └── unittest/
│   │       └── unitTest.py
```

Cela facilite la navigation dans le projet, la compréhension de la structure du code, et rend le travail en équipe plus harmonieux.
