import cv2

# Initialize the camera
cam = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cam.isOpened():
    print("Unable to open the camera.")
    exit()

# Capture a single frame
ret, frame = cam.read()

# If the frame is captured without any error, save it
if ret:
    cv2.imwrite("captured_image.png", frame)
    print("Image saved successfully.")
else:
    print("Failed to capture the image.")

# Release the camera
cam.release()
