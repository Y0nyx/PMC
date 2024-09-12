import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from shapely.affinity import scale

def yolo_to_pixels(coord, image_width, image_height):
    pixels = []
    for i in range(0, len(coord), 2):
        x = int(coord[i] * image_width)
        y = int(coord[i+1] * image_height)
        pixels.append((x, y))
    return pixels

def create_segmentation_mask(yolo_coords, image_width, image_height):
    polygons = []
    for coord in yolo_coords:
        pixels = yolo_to_pixels(coord, image_width, image_height)
        polygons.append(Polygon(pixels))
    return polygons

def polygon_to_mask(polygons, image_width, image_height):
    x = np.arange(image_width)
    y = np.arange(image_height)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.flatten(), yy.flatten())).T
    mask = np.zeros((image_height, image_width))
    for polygon in polygons:
        mask_points = np.array(polygon.contains_points(points), dtype=int).reshape(image_height, image_width)
        mask[mask_points == 1] = 1
    return mask

chemin_parent = 'D:\\dataset\\welding-detection\\valid'
images_path = os.path.join(chemin_parent, 'images')
labels_path = os.path.join(chemin_parent, 'labels')

for img_name in os.listdir(images_path):
    name = os.path.splitext(img_name)[0]
    img_path = os.path.join(images_path, img_name)
    label_path = os.path.join(labels_path, name + ".txt")

    # Charger l'image
    img = Image.open(img_path)
    image_width, image_height = img.size

    # Charger les coordonnées YOLO à partir du fichier de segmentation
    with open(label_path, 'r') as file:
        lines = file.readlines()
        yolo_coords = []
        for line in lines:
            yolo_coords.append([float(coord) for coord in line[1:].strip().split()])
    # Créer le masque de segmentation
    polygons = create_segmentation_mask(yolo_coords, image_width, image_height)

    # Conversion du masque en masque numpy
    mask = polygon_to_mask(polygons, image_width, image_height)

    # Appliquer le masque à l'image
    masked_img = np.array(img)
    masked_img[:, :, 0] = np.where(mask == 1, masked_img[:, :, 0], 0)
    masked_img[:, :, 1] = np.where(mask == 1, masked_img[:, :, 1], 0)
    masked_img[:, :, 2] = np.where(mask == 1, masked_img[:, :, 2], 0)

    # Afficher l'image avec les pixels de la segmentation
    plt.imshow(masked_img)
    plt.axis('off')
    plt.show()
