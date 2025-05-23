# PMC
## Commencer à coder 
Il faut s'assurer que le requirements.txt en utilisant la commande

`pip install -r requirements.txt`

** Il faut essayer d'éviter d'ajouter des librairie pour rendre notre code le plus léger.

Pour mettre le requirements a jours, veiller utiliser la librairie pipreqs avec cette commande. Cela va scanner le projet entièrement et rajouter les librairie nécessaire.

`pipreqs /home/project/location`

# Test Unitaire
## Faire un test unitaire
Pytest va aller chercher tout fichier commençant par 'test_<nom du fichier tester>.py et donc suivre cette nomenclature pour le développement des unittests. Tout les testes se trouve dans src/test. Suivre les exemples de test déjà fait et utiliser tout AI de génération de code pour vous aider dans le développement de test.

## Lancer les tests unitaire
Exécuter cette commande au root du projet

`python -m pytest`

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
