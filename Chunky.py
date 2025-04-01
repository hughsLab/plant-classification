import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# Parameters
batch_size = 32
img_height = 128
img_width = 128
initial_epochs_per_chunk = 20  # Starting epochs per chunk
num_classes = 10001  # Adjust based on your actual class count

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
    r"D:\PLANT\Chunkus\CHUNK_1",
       r"D:\PLANT\Chunkus\CHUNK_2", 
                r"D:\PLANT\Chunkus\CHUNK_3",
                 r"D:\PLANT\Chunkus\CHUNK_4",
                  r"D:\PLANT\Chunkus\CHUNK_5",
                   r"D:\PLANT\Chunkus\CHUNK_6",
                     r"D:\PLANT\Chunkus\CHUNK_7",
                      r"D:\PLANT\Chunkus\CHUNK_8",
                     r"D:\PLANT\Chunkus\CHUNK_9",
                        r"D:\PLANT\Chunkus\CHUNK_10",
]

# Initialize and compile the model with default Adam optimizer learning rate
model = create_model(num_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Loop through each chunk, gradually freezing layers and reducing epochs
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

    # Configure performance with caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Freeze early layers after the first chunk
    if chunk_index > 0:
        for layer in model.layers[:4]:  # Freeze first 4 layers
            layer.trainable = False

    # Adjust epochs per chunk (reduce by 5 for each new chunk)
    current_epochs = max(initial_epochs_per_chunk - chunk_index * 5, 10)
    
    # Train the model on this chunk
    history = model.fit(train_ds, validation_data=val_ds, epochs=current_epochs)
    
    # Save model weights after each chunk
    model.save_weights(f'D:\\PLANT\\SavedWeights\\model_weights_chunk{chunk_index}.h5')

    # Unfreeze all layers for the next chunk
    for layer in model.layers:
        layer.trainable = True

# Visualization of final training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(current_epochs)

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

# After training on all chunks, you can predict on new images
test_images = [
    r"D:\PLANT\Chunk1.jpg",
      r"D:\PLANT\Chunk2.jpg",
    r"D:\PLANT\Chunk3.jpg",
    r"D:\PLANT\Chunk4.jpg",
    r"D:\PLANT\Chunk5.jpg",
    r"D:\PLANT\Chunk6.jpg",
    r"D:\PLANT\Chunk7.jpg",
    r"D:\PLANT\Chunk8.jpg",
    r"D:\PLANT\Chunk9.jpg",
   r"D:\PLANT\Chunk10.jpg",
]

for flower_path in test_images:
    img = tf.keras.utils.load_img(
        flower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        f"This image {flower_path} most likely belongs to class '{class_names[np.argmax(score)]}' "
        f"with a {100 * np.max(score):.2f} percent confidence."
    )
