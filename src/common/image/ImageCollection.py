from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class ImageCollection:
    def __init__(self, img_list=None) -> None:
        """
        Initialise un objet ImageCollection.

        Paramètres :
        - img_list (list): Liste d'objets image. Par défaut, None.
        """
        self._img_list = img_list if img_list is not None else []
        self._save_counter = 0

    def __iter__(self):
        """
        Permet l'itération sur la collection d'images.
        """
        return iter(self._img_list)

    @property
    def shape(self):
        """
        Récupère la taille de chaque image dans la collection.

        Renvoie :
        - list: Liste de tuples contenant la taille de chaque image.
        """
        return [img.shape() for img in self._img_list]

    def resize(self, width, height) -> bool:
        """
        Redimensionne toutes les images de la collection aux dimensions spécifiées.

        Paramètres :
        - width (int): Nouvelle largeur des images.
        - height (int): Nouvelle hauteur des images.

        Renvoie :
        - bool: True si toutes les images ont été redimensionnées avec succès, False sinon.
        """
        return all(img.resize(width, height) for img in self._img_list)

    def save(self, file_path: Path):
        """
        Enregistre toutes les images de la collection dans un répertoire spécifié.

        Paramètres :
        - file_path (Path): Chemin vers le répertoire où les images seront enregistrées.
        """
        os.makedirs(file_path, exist_ok=True)
        self._save_counter = len(os.listdir(file_path))
        for i, img in enumerate(self._img_list, start=self._save_counter):
            img.save(file_path / f"img_{i}.png")
        self._save_counter += len(self._img_list)

    def add(self, img):
        """
        Ajoute une image à la collection.

        Paramètres :
        - img: Objet image à ajouter.
        """
        self._img_list.append(img)
