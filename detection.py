import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image
image_path = "airplane.jpg"  # Replace with the actual path to your image
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Initialize the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Perform object detection
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()

# Class labels for MobileNet SSD
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Iterate through detections and print class and confidence for all
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    class_id = int(detections[0, 0, i, 1])

    # Print class name and confidence for each detection
    class_name = classes[class_id] if class_id < len(classes) else "unknown"
    print(f"Detected: {class_name}, Confidence: {confidence:.2f}")

    # Check if the detected object has confidence above a threshold
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw rectangle and label
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image using cv2_imshow
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
