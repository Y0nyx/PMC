import os
import shutil

def copy_images_from_selected_cameras(src_directory, dest_directory, cameras):
    # List all files in the source directory
    files = os.listdir(src_directory)

    # Filter files with filename ending with the specified cameras and common image extensions
    image_files = []
    for camera in cameras:
        # Use os.path.splitext to get the file extension and compare it with allowed extensions
        allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        image_files += [file for file in files if file.lower().endswith(tuple(str(camera) + ext for ext in allowed_extensions))]

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_directory, exist_ok=True)

    # Copy the selected image files to the destination directory
    for image in image_files:
        src_path = os.path.join(src_directory, image)
        dest_path = os.path.join(dest_directory, image)
        shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    # Provide the source and destination directory paths
    source_directory_path = "captured_images_1_2_3"
    destination_directory_path = "dataset_selected_cam_images"
    
    #Provide the camera selection
    cameras = [1]
    
    # Copy images with filename ending with '1' and '2' to the destination directory
    copy_images_from_selected_cameras(source_directory_path, destination_directory_path, cameras)
    
    print("The images have been copied to the destination directory.")
