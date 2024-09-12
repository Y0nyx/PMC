from PIL import Image
import os

def make_divisible_by_subdivisor(value, subdivisor):
    # Calculate the adjustment required to make the value divisible by subdivisor
    remainder = value % subdivisor
    adjustment = subdivisor - remainder if remainder != 0 else 0
    return value + adjustment

def crop_and_save_images(input_folder, output_folder, cropping_specs):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Check if the file is an image (you can add more extensions if needed)
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Load the image
            image_path = os.path.join(input_folder, file)
            image = Image.open(image_path)

            for idx, crop_spec in enumerate(cropping_specs, start=1):
                # Unpack the cropping specification
                crop_left, crop_top, crop_right, crop_bottom = crop_spec["coordinates"]
                crop_width = crop_right - crop_left
                crop_height = crop_bottom - crop_top

                # Calculate adjustments to make dimensions divisible by subdivisor
                crop_width_adjustment = make_divisible_by_subdivisor(crop_width, crop_spec["subdivisor"]) - crop_width
                crop_height_adjustment = make_divisible_by_subdivisor(crop_height, crop_spec["subdivisor"]) - crop_height

                # Apply adjustments
                crop_right += crop_width_adjustment/2
                crop_bottom += crop_height_adjustment/2
                crop_left -= crop_width_adjustment/2
                crop_top -= crop_height_adjustment/2

                # Crop and save the version
                cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
                output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_cropped_{idx}{os.path.splitext(file)[1]}")
                cropped.save(output_path)

                print(f"Processed: {file}, Cropped Version: {idx}")

if __name__ == "__main__":
    # Provide the input and output folder paths
    input_folder_path = "dataset_selected_cam_images"
    output_folder_path = "dataset_cropped_images"
    subdivisor = 128
    
    # Define cropping specifications
    cropping_specs = [
        {"coordinates": (806, 479, 1275, 814), "subdivisor": subdivisor},
        {"coordinates": (1224, 682, 1804, 1057), "subdivisor": subdivisor},
        # Add more cropping specifications as needed
    ]
    
    # Crop and save images
    crop_and_save_images(input_folder_path, output_folder_path, cropping_specs)
    
    print("Images have been cropped and saved to the output folder.")
