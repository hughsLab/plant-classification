import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="https://app.roboflow.com/ds/sNzfzskhZZ?key=yOATKm56Or", epochs=75, imgsz=640)
# Define the path to your input image
input_image_path = r"D:\PLANT\Chunkus\Chunk_1\Abelmoschus_esculentus_(L.)_Moench_Abelmoschus_Malvaceae\121237.jpg"

# Run object detection
print("Running object detection...")
results = model(input_image_path)  # Runs inference

# Load the input image
image = cv2.imread(input_image_path)

# Iterate through the results and draw bounding boxes
for result in results:
    for box in result.boxes:
        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Optionally, you can put a label on the bounding box
        label = box.conf[0]  # Confidence score
        cv2.putText(image, f'{label:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Output Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()