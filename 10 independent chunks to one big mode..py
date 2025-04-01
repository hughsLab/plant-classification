import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import pickle

# Parameters
batch_size = 32
img_height = 128
img_width = 128
initial_epochs_per_chunk = 45  # Starting epochs per chunk

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Model creation function
def create_model(num_classes):
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),  # Dropout layer to reduce overfitting
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])
    return model

# Set the paths to your dataset chunks
chunk_paths = [
    r"D:\PLANT\Chunkus\CHUNK_1", r"D:\PLANT\Chunkus\CHUNK_2", 
    r"D:\PLANT\Chunkus\CHUNK_3", r"D:\PLANT\Chunkus\CHUNK_4",
    r"D:\PLANT\Chunkus\CHUNK_5", r"D:\PLANT\Chunkus\CHUNK_6",
    r"D:\PLANT\Chunkus\CHUNK_7", r"D:\PLANT\Chunkus\CHUNK_8",
    r"D:\PLANT\Chunkus\CHUNK_9", r"D:\PLANT\Chunkus\CHUNK_10",
]

# Dictionary to save class names for each chunk and track total classes
chunk_class_names = {}
total_classes = 0  # Total number of unique classes across all chunks

# Train each chunk separately
for chunk_index, chunk_path in enumerate(chunk_paths):
    data_dir = pathlib.Path(chunk_path)
    if not data_dir.exists():
        print(f"The directory {data_dir} does not exist.")
        continue

    # Load the dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)  # Dynamically set num_classes per chunk
    chunk_class_names[chunk_index] = class_names  # Save class names for each chunk
    total_classes += num_classes

    # Rebuild the model for each chunk to reset output classes
    model = create_model(num_classes)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Configure performance with caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Train the model on this chunk
    history = model.fit(train_ds, validation_data=val_ds, epochs=initial_epochs_per_chunk)
    
    # Save model weights and class names after each chunk
    model.save_weights(f'D:\\PLANT\\SavedWeights\\model_weights_chunk{chunk_index}.weights.h5')

# Save the class names dictionary
with open('D:\\PLANT\\SavedWeights\\class_names.pkl', 'wb') as f: 
    pickle.dump(chunk_class_names, f)

# Combine models into a single big model for all classes
big_model = create_model(total_classes)
big_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# Test each image using the combined model
test_images = [
    (r"D:\PLANT\Chunk1.jpg", 0),
    (r"D:\PLANT\Chunk2.jpg", 1),
    (r"D:\PLANT\Chunk3.jpg", 2),
    (r"D:\PLANT\Chunk4.jpg", 3),
    (r"D:\PLANT\Chunk5.jpg", 4),
    (r"D:\PLANT\Chunk6.jpg", 5),
    (r"D:\PLANT\Chunk7.jpg", 6),
    (r"D:\PLANT\Chunk8.jpg", 7),
    (r"D:\PLANT\Chunk9.jpg", 8),
    (r"D:\PLANT\Chunk10.jpg", 9),
]

class_names_combined = sum(chunk_class_names.values(), [])  # Flatten all class names

for flower_path, chunk_index in test_images:
    # Prepare the test image
    img = tf.keras.utils.load_img(
        flower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions with the big model
    predictions = big_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        f"This image {flower_path} most likely belongs to class '{class_names_combined[np.argmax(score)]}' "
        f"with a {100 * np.max(score):.2f} percent confidence."
    )
