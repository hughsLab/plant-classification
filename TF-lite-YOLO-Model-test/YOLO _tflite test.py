from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("yolo11n_saved_model/yolo11n_float32.tflite")

# Run inference
results = tflite_model("D:\PLANT\TEST_TRAIN_1\Abeliophyllum_distichum_Nakai_Abeliophyllum_Oleaceae\bus.jpg")


# Define the path to your input image
input_image_path = r"D:\PLANT\TEST_TRAIN_1\Abeliophyllum_distichum_Nakai_Abeliophyllum_Oleaceae\bus.jpg"

# Run object detection
print("Running object detection...")
results = model(input_image_path)  # Runs inference
