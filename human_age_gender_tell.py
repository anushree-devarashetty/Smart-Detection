import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Load the image
image_path = "trio.jpg"  # Replace with the actual path to your image
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert the image to RGB (MTCNN uses RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create the MTCNN detector
detector = MTCNN()

# Perform face detection
faces = detector.detect_faces(image_rgb)

# Draw rectangles around detected faces
for face in faces:
    x, y, width, height = face['box']
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Extract the face region for age and gender estimation
    face_roi = image[y:y+height, x:x+width]

    # Perform age and gender estimation (replace with your preferred method)
    # age, gender = your_age_gender_estimation_method(face_roi)
    age, gender = 25, "Female"  # Replace with actual values or estimation method

    # Display age and gender on the image
    text = f"Age: {age}, Gender: {gender}"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image using matplotlib
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()
