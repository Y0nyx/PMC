from PIL import Image
import os

def add_overlap( left, top, right, bottom, width, height, overlap_size):
    # Add overlap pixels from the original image on each side
        left -= overlap_size
        top -= overlap_size
        right += overlap_size
        bottom += overlap_size
        if(left < 0):
            left = 0
            right+=overlap_size
        if(right > width):
            right = width
            left -= overlap_size
        
        if(top < 0):
            top = 0
            bottom+=overlap_size
        if(bottom > height):
            bottom = height
            top -=overlap_size
            
        return left, top, right, bottom

def create_sub_images(input_folder, output_folder, sub_image_size, overlap_size, transformation_type):
    """
    Create sub-images from images in the input folder with specified parameters.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where sub-images will be saved.
        sub_image_size (int): Size of each sub-image (width and height).
        overlap_size (int): Size of the overlap (in pixels) between sub-images.
        transformation_type (str): Type of transformation to apply to sub-images.
    """
if __name__ == '__main__':
    # Provide the input and output folder paths
    input_folder_path = "dataset_cropped_images"
    output_folder_path_overlap = "dataset_sub_images"
    output_folder_path_translated_horizontal = "dataset_sub_images_horizontal_translation"
    output_folder_path_translated_vertical = "dataset_sub_images_vertical_translation"
    sub_image_size = 128  # Size of the sub-images
    overlap_size = 64  # Size of the overlap

    # Create sub-images with overlap
    create_sub_images(input_folder_path, output_folder_path_overlap, sub_image_size, overlap_size, "untranslated")

    # Create translated sub-images (horizontal)
    create_sub_images(input_folder_path, output_folder_path_translated_horizontal, sub_image_size, overlap_size, "translated_horizontal")

    # Create translated sub-images (vertical)
    create_sub_images(input_folder_path, output_folder_path_translated_vertical, sub_image_size, overlap_size, "translated_vertical")

    print("All images have been subdivised with overlapping pixels")
