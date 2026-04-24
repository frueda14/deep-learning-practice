import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import kagglehub
import os

# 1. DOWNLOAD AND PATH SETUP
# -----------------------------------------------------------
path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

# Folder structure: path + 'horse-or-human' + 'train/validation'
train_dir = os.path.join(path, 'horse-or-human', 'train')
validation_dir = os.path.join(path, 'horse-or-human', 'validation')

print(f"Training path: {train_dir}")
print(f"Validation path: {validation_dir}")

# 2. DATA GENERATORS
# -----------------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1.0/255.)
val_datagen  = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary' # Binary classification exercise requirement
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 3. LeNet-5 MODEL (Adapted for binary classification)
# -----------------------------------------------------------
# Following the reference LeNet-5 architecture
model = models.Sequential([
    # Layer 1: 6 filters + Pooling
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(64, 64, 3)),
    layers.AveragePooling2D(),

    # Layer 2: 16 filters + Pooling
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.AveragePooling2D(),

    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),

    # BINARY OUTPUT: single unit with Sigmoid activation
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy', # Required for binary classification
              metrics=['accuracy'])

# 4. TRAINING
# -----------------------------------------------------------
print("\nStarting training! 🚀")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 5. PLOTS (Performance visualization)
# -----------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.show()
