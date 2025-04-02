import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import pathlib

# Set the local path to your folder
data_dir = pathlib.Path(r"D:\PLANT\Minitest")

# Check if the directory exists
if not data_dir.exists():
    print(f"The directory {data_dir} does not exist.")
else:
    # Count total images
    image_count = len(list(data_dir.glob('**/*.jpg')))
    print(f"Total images: {image_count}")

# Load data using a Keras utility
batch_size = 64
img_height = 128
img_width = 128

# Load the training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

# Standardize the data
normalization_layer = layers.Rescaling(1./255)

# Define a function to process data (normalization and label processing)
def process_data(image, label):
    label_one_hot = tf.one_hot(label, num_classes)  # Shape: [batch_size, num_classes]
    bbox_constant = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)  # Shape: [1, 4]
    bbox_tiled = tf.tile(bbox_constant, [tf.shape(label)[0], 1])  # Match batch size
    label_combined = tf.concat([label_one_hot, bbox_tiled], axis=1)  # Concatenate along axis 1
    return image, label_combined


# Apply the data processing to the datasets
train_ds = train_ds.map(process_data).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(process_data).cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Model with bounding box output
model = Sequential([
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes + 4, name="outputs")  # +4 for bounding box (x_min, y_min, x_max, y_max)
])

# Custom loss function
def custom_loss(y_true, y_pred):
    # Separate class and bounding box predictions
    y_true_class, y_true_bbox = y_true[:, :num_classes], y_true[:, -4:]
    y_pred_class, y_pred_bbox = y_pred[:, :num_classes], y_pred[:, -4:]

    # Classification loss
    class_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true_class, y_pred_class)

    # Bounding box regression loss (Mean Squared Error)
    bbox_loss = tf.keras.losses.MeanSquaredError()(y_true_bbox, y_pred_bbox)

    return class_loss + bbox_loss

# Compile the model
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

# Train the model
epochs = 70  # Adjust for testing purposes
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Save and visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Convert the trained model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "plant_detection_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"Model saved to {tflite_model_path}")

# Example prediction on a new image
flower_path = r"D:\PLANT\Chunk1.jpg"

img = tf.keras.utils.load_img(flower_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = normalization_layer(tf.expand_dims(img_array, 0))  # Normalize the image

predictions = model.predict(img_array)
class_probs = tf.nn.softmax(predictions[0, :-4])
bbox_coords = predictions[0, -4:]  # Bounding box: [x_min, y_min, x_max, y_max]

predicted_class = class_names[tf.argmax(class_probs).numpy()]
confidence = tf.reduce_max(class_probs).numpy()
print(f"This image most likely belongs to {predicted_class} with a {confidence:.2f} confidence.")
print(f"Bounding box coordinates (relative): {bbox_coords}")
