from ultralytics import YOLO
import cv2
import numpy as np
import os

# Define the correct model path
model_path = r"C:\Users\USER\runs\detect\train\weights\best.pt"

# Verify if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")

# Load the trained model
model = YOLO(model_path)



# Export the model to TensorFlow Lite format
print("Exporting model to TFLite format...")
tflite_model_path = model.export(format="tflite")  # creates 'yolo11n_float32.tflite'






# Verify export success
if tflite_model_path:
    print(f"Model successfully exported to {tflite_model_path}")
else:
    print("Model export failed.")
    exit()

# Define the path to your input image
input_image_path = r"D:\PLANT\bus.jpg"

# Run object detection
print("Running object detection...")
results = model(input_image_path)  # Runs inference

# Extract detection results
for result in results:
    img = result.orig_img  # Get the original image with detections

    # Draw bounding boxes on the image
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        confidence = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class index

        # Draw rectangle and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        label = f"Class: {cls}, Conf: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nObject detection completed!")
