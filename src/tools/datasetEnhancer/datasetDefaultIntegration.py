import argparse
import os
import random
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
import matplotlib.pyplot as plt
import numpy as np

def fit_square_inside_polygon(segmentation_polygon, square_polygon, scale_factor=0.99, pourcent=80, show=False, min_size=0.1):
    while True:
        intersection_area = square_polygon.intersection(segmentation_polygon).area

        # Calculate the percentage of the square covered by the polygon
        percent_covered = (intersection_area / square_polygon.area) * 100
        if percent_covered < pourcent:
            if (square_polygon.bounds[2] - square_polygon.bounds[0]) < min_size or (square_polygon.bounds[3] - square_polygon.bounds[1]) < min_size:
                break
            square_polygon = scale(square_polygon, xfact=scale_factor, yfact=scale_factor)
        else:
            break

    # Créer une figure Matplotlib
    if show:
        _ , ax = plt.subplots()

        # Dessiner le premier polygone en bleu avec une transparence de 0.5
        ax.fill(*square_polygon.exterior.xy, color='blue', alpha=0.5)

        # Dessiner le deuxième polygone en rouge avec une transparence de 0.5
        ax.fill(*segmentation_polygon.exterior.xy, color='red', alpha=0.5)

        # Définir les limites de l'axe
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-0.5, 2)

        # Afficher la figure
        plt.show()

    return square_polygon

def get_polygon(nom_fichier):
    with open(nom_fichier, 'r') as f:
        for ligne in f:
            ligne = ligne.split(' ')
            coordonnees = [float(l) for l in ligne]
    segmentation_coords = coordonnees[1:]
    points = [(segmentation_coords[i], segmentation_coords[i+1]) for i in range(0, len(segmentation_coords), 2)]
    return Polygon(points)

def get_points(polygon, nbr: int = 100):
    points = []
    while len(points) < nbr:
        # Generate a random point within the bounding box of the polygon
        random_point = Point(np.random.uniform(polygon.bounds[0], polygon.bounds[2]),
                            np.random.uniform(polygon.bounds[1], polygon.bounds[3]))
        # Check if the point is inside the polygon
        if random_point.intersects(polygon):
            points.append(random_point)
    
    return points

def add_defect(image, segmentation, defect_image, points, show):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    defect_image = cv2.cvtColor(defect_image, cv2.COLOR_RGB2GRAY)

    point = random.choice(points)
    x = point.x
    y = point.y

    width = defect_image.shape[0] / image.shape[0]
    length = width = defect_image.shape[1] / image.shape[1]

    square_polygon = Polygon([
        (x - width/2 , y - length/2),
        (x + width/2, y - length/2),
        (x + width/2, y + length/2),
        (x - width/2, y + length/2)
    ])

    fitted_square = fit_square_inside_polygon(segmentation, square_polygon, show=show)

    width = (fitted_square.bounds[2] - fitted_square.bounds[0]) * image.shape[0]
    height = (fitted_square.bounds[3] - fitted_square.bounds[1]) * image.shape[1]

    x = int(x * image.shape[1])
    y = int(y * image.shape[0])

    # Récupérer la région d'intérêt (ROI) de l'image d'origine
    roi = image[y:y+int(width), x:x+int(height)]

    defect_resized = cv2.resize(defect_image, (roi.shape[1], roi.shape[0]))
    # Méthode 1 : Histogram Equalization
    defect_resized = cv2.equalizeHist(defect_resized)

    final_roi = cv2.addWeighted(roi, 0.5, defect_resized, 0.5, 0)

    # Remplacer la région d'intérêt de l'image principale par l'image de défaut redimensionnée
    image[y:y+defect_resized.shape[0], x:x+defect_resized.shape[1]] = final_roi

    #image = cv2.equalizeHist(image)

    # Calculer les coordonnées de la bounding box
    x_center = (x + (x + defect_resized.shape[1])) / 2 / image.shape[1]
    y_center = (y + (y + defect_resized.shape[0])) / 2 / image.shape[0]
    width = defect_resized.shape[1] / image.shape[1]
    height = defect_resized.shape[0] / image.shape[0]
    if show:
        plt.imshow(image, cmap='gray')
        plt.title('Image with defect')
        plt.show()

    return image, (x_center, y_center, width, height)

def draw_bbox_yolo(image, bbox_yolo):
    """
    Dessine une boîte englobante sur l'image à partir des coordonnées YOLO.
    
    Args:
    - image : image sur laquelle dessiner la boîte englobante
    - bbox_yolo : liste contenant les coordonnées de la boîte englobante au format YOLO (center_x, center_y, width, height)
    
    Returns:
    - image : image avec la boîte englobante dessinée
    """
    width, height = image.shape[1], image.shape[0]
    center_x, center_y, bbox_width, bbox_height = bbox_yolo
    
    # Convertir les coordonnées YOLO en coordonnées de coin supérieur gauche et inférieur droit
    x_min = int((center_x - bbox_width/2) * width)
    y_min = int((center_y - bbox_height/2) * height)
    x_max = int((center_x + bbox_width/2) * width)
    y_max = int((center_y + bbox_height/2) * height)
    
    # Dessiner la boîte englobante sur l'image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return image


def main(input_folder, output_folder, defect_folder, defect_percentage, show):
    # Charger les images de soudure
    input_images = os.listdir(os.path.join(input_folder, 'images'))

    # Charger les images de défaut
    defect_images = [cv2.imread(os.path.join(defect_folder, img)) for img in os.listdir(defect_folder)]

    # Créer un dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)

    # Boucle sur chaque image de soudure avec tqdm
    for image_name in tqdm(input_images, desc="Processing images"):
        # Charger l'image de soudure
        image = cv2.imread(os.path.join(input_folder, 'images', image_name))

        # Initialiser la liste pour stocker les coordonnées des bounding boxes
        bounding_boxes = []

        # Vérifier si un défaut doit être ajouté à cette image
        if random.random() < defect_percentage:
            # Choisir un défaut aléatoire parmi les images de défaut
            defect_image = random.choice(defect_images)

            # Ajouter le défaut à l'image de soudure
            segmentation = get_polygon(os.path.join(input_folder, 'labels', os.path.splitext(image_name)[0] + '.txt'))
            points = get_points(segmentation)
            image_with_defect, bbox = add_defect(image, segmentation, defect_image, points, show)

            if bbox:
                bounding_boxes.append(bbox)

            # Afficher l'image avec la boîte englobante si show est True
            if show:
                image_with_bbox = draw_bbox_yolo(image_with_defect, bbox)
                plt.imshow(image_with_bbox, cmap='gray')
                plt.title('Image with bbox')
                plt.show()

        # Enregistrer l'image originale
        output_path = os.path.join(output_folder, 'images', image_name)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

        # Créer un fichier vide pour les labels
        with open(os.path.join(output_folder, 'labels', os.path.splitext(image_name)[0] + ".txt"), "w") as f:
            pass

        # Enregistrer l'image avec le défaut et écrire les coordonnées des bounding boxes
        if bounding_boxes:
            output_path = os.path.join(output_folder, 'images', os.path.splitext(image_name)[0] + "_defect.jpg")
            cv2.imwrite(output_path, image_with_defect)

            # Écrire les coordonnées des bounding boxes dans un fichier texte au format YOLO
            with open(os.path.join(output_folder, 'labels', os.path.splitext(image_name)[0] + "_defect.txt"), "w") as f:
                for bbox in bounding_boxes:
                    f.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(bbox[0], bbox[1], bbox[2], bbox[3]))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images and add defects.')
    parser.add_argument('--input_folder', type=str, default='D:\\dataset\\v10i.yolov8', help='Input folder containing the original images')
    parser.add_argument('--output_folder', type=str, default='D:\\dataset\\dataset_with_equalize', help='Output folder to save the processed images')
    parser.add_argument('--defect_folder', type=str, default='D:\\dataset\\v5i.yolov8', help='Folder containing defect images')
    parser.add_argument('--defect_percentage', type=float, default=0.5, help='Percentage of images with defects')
    parser.add_argument('--show', action='store_true', help='Display images with bounding boxes')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    folders = ['train', 'valid', 'test']
    for folder in folders:
        input_folder = os.path.join(args.input_folder, folder)
        output_folder = os.path.join(args.output_folder, folder)
        defect_folder = os.path.join(args.defect_folder, folder + '\\images')
        show = args.show

        main(input_folder, output_folder, defect_folder, args.defect_percentage, show)
