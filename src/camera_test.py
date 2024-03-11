import cv2

def list_cameras():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            print(f"Camera {index} is available.")
            cap.release()
        index += 1

list_cameras()

# Initialize the camera
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(2)
cam2 = cv2.VideoCapture(4)
cam3 = cv2.VideoCapture(6)

camlist = [cam0, cam1, cam2, cam3]

for i, cam in enumerate(camlist):
    # Check if the camera is opened correctly
    if not cam.isOpened():
        print("Unable to open the camera.")
        exit()

    # Capture a single frame
    ret, frame = cam.read()

    # If the frame is captured without any error, save it
    if ret:
        filename = f"captured_image_cam{i}.png"
        cv2.imwrite(filename, frame)
        print("Image saved successfully.")
    else:
        print("Failed to capture the image.")

    # Release the camera
    cam.release()
