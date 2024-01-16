from PIL import Image
import os

def rotate_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Loop through each image file and perform rotations
    for image_file in image_files:
        # Construct the full path for the input and output images
        input_path = os.path.join(input_folder, image_file)
        output_base_name = os.path.splitext(image_file)[0]

        # Open the image
        with Image.open(input_path) as img:
            # Rotate and save the image for each rotation angle
            for angle in [0, 90, 180, 270]:
                rotated_img = img.rotate(angle)
                output_path = os.path.join(output_folder, f"{output_base_name}_rotated_{angle}.png")
                rotated_img.save(output_path)

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "images"
    output_folder = "dataset_rotated_images"

    # Call the function to rotate images
    rotate_images(input_folder, output_folder)
