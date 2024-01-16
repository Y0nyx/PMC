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
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Check if the file is an image (you can add more extensions if needed)
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Load the imagey
            image_path = os.path.join(input_folder, file)
            original_image = Image.open(image_path)

            # Apply the specified transformation
            if transformation_type == "untranslated":
                create_untranslated_sub_images(original_image, output_folder, sub_image_size, overlap_size, file)
            elif transformation_type == "translated_horizontal":
                create_translated_sub_images(original_image, output_folder, sub_image_size, overlap_size, file, axis="horizontal")
            elif transformation_type == "translated_vertical":
                create_translated_sub_images(original_image, output_folder, sub_image_size, overlap_size, file, axis="vertical")

def create_untranslated_sub_images(original_image, output_folder, sub_image_size, overlap_size, file):
    # Get the dimensions of the original image
    width, height = original_image.size

    # Calculate the number of sub-images in both dimensions
    num_sub_images_x = width // sub_image_size
    num_sub_images_y = height // sub_image_size

    # Iterate over the sub-images and save each one with overlap
    for i in range(num_sub_images_x):
        for j in range(num_sub_images_y):
            left = i * sub_image_size
            top = j * sub_image_size
            right = left + sub_image_size
            bottom = top + sub_image_size

            left, top, right, bottom = add_overlap(left, top, right, bottom, width, height, overlap_size)

            # Crop and save the sub-image with overlap
            sub_image = original_image.crop((left, top, right, bottom))
            output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_sub_{i}_{j}_overlap.png")
            sub_image.save(output_path)

def create_translated_sub_images(original_image, output_folder, sub_image_size, overlap_size, file, axis):
    # Get the dimensions of the original image
    width, height = original_image.size

    # Calculate the number of sub-images in both dimensions
    num_sub_images_x = width // sub_image_size
    num_sub_images_y = height // sub_image_size

    if(axis=="horizontal"): num_sub_images_x-=1
    else: num_sub_images_y-=1

    # Iterate over the sub-images and save each one with overlap
    for i in range(num_sub_images_x):
        for j in range(num_sub_images_y):
            left = i * sub_image_size
            top = j * sub_image_size
            right = left + sub_image_size
            bottom = top + sub_image_size

            # Add overlap pixels from the original image based on the specified axis
            if axis == "horizontal":
                left += sub_image_size/2
                right += sub_image_size/2
            elif axis == "vertical":
                top += sub_image_size/2
                bottom += sub_image_size/2
            else:
                raise ValueError("Invalid axis. Use 'horizontal' or 'vertical'.")

            left, top, right, bottom = add_overlap(left, top, right, bottom, width, height, overlap_size)

            # Crop and save the sub-image with overlap
            sub_image = original_image.crop((left, top, right, bottom))
            output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_translated_{axis}_{i}_{j}_overlap.png")
            sub_image.save(output_path)

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