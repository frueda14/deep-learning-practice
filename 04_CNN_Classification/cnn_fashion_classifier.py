import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print(f"TensorFlow Version: {tf.__version__}")

# 1. LOAD DATA (Fashion MNIST)
# ---------------------------------------------------------
# Classes: 0:T-shirt/top, 1:Trouser, 2:Pullover, 3:Dress, 4:Coat,
# 5:Sandal, 6:Shirt, 7:Sneaker, 8:Bag, 9:Ankle boot
print("Loading Fashion MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape: CNNs expect (Batch, Height, Width, Channels)
# Grayscale images have 1 channel.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")

# 2. BUILD THE CNN (Simplified "VGG-style" architecture)
# ---------------------------------------------------------
model = models.Sequential([
    # Convolutional Block 1: detects edges and simple textures
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(), # Speeds up training and improves stability
    layers.MaxPooling2D((2, 2)), # Downsamples to 14x14

    # Convolutional Block 2: detects more complex shapes (sleeves, heels)
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), # Downsamples to 7x7

    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)), # Downsamples to 3x3

    # Flatten and classify
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4), # Randomly drops 40% of neurons to prevent overfitting
    layers.Dense(10, activation='softmax') # 10 neurons = 10 clothing categories
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. TRAINING (With Early Stopping)
# ---------------------------------------------------------
# Stop training if val_loss does not improve for 5 consecutive epochs.
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\nStarting training... 🚀")
history = model.fit(x_train, y_train,
                    epochs=20, # Up to 20 epochs; early stopping will likely trigger first
                    batch_size=64,
                    validation_split=0.2, # Reserve 20% of training data for validation
                    callbacks=[early_stopping])

# 4. EVALUATION AND VISUALIZATION
# ---------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Learning curves
plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# 5. CONFUSION MATRIX (Which clothing items get confused?)
# ---------------------------------------------------------
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
