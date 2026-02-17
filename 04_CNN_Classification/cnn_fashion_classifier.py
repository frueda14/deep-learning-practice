import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuración para que se vea bonito en VS Code
print(f"TensorFlow Version: {tf.__version__}")

# 1. CARGAR DATOS (Fashion MNIST)
# ---------------------------------------------------------
# Clases: 0:T-shirt/top, 1:Trouser, 2:Pullover, 3:Dress, 4:Coat,
# 5:Sandal, 6:Shirt, 7:Sneaker, 8:Bag, 9:Ankle boot
print("Cargando dataset Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalizar los valores de píxeles (de 0-255 a 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Redimensionar: Las CNN esperan (Batch, Alto, Ancho, Canales)
# Como es escala de grises, el canal es 1.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Forma de entrenamiento: {x_train.shape}")
print(f"Forma de test: {x_test.shape}")

# 2. CONSTRUIR LA CNN (Arquitectura "VGG-style" simplificada)
# ---------------------------------------------------------
model = models.Sequential([
    # Bloque Convolucional 1: Detecta bordes y texturas simples
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(), # Ayuda a entrenar más rápido y estable
    layers.MaxPooling2D((2, 2)), # Reduce a 14x14
    
    # Bloque Convolucional 2: Detecta formas más complejas (mangas, tacones)
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), # Reduce a 7x7
    
    # Bloque Convolucional 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)), # Reduce a 3x3
    
    # Aplanar y Clasificar
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4), # Apaga el 40% de neuronas al azar para evitar memorización (Overfitting)
    layers.Dense(10, activation='softmax') # 10 neuronas = 10 tipos de ropa
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. ENTRENAMIENTO (Con Early Stopping)
# ---------------------------------------------------------
# Si el 'val_loss' no mejora en 5 épocas, paramos para no perder tiempo.
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

print("\nIniciando entrenamiento... 🚀")
history = model.fit(x_train, y_train,
                    epochs=20, # Le damos hasta 20, pero el early_stopping probablemente pare antes
                    batch_size=64,
                    validation_split=0.2, # Usamos 20% de train para validar mientras entrena
                    callbacks=[early_stopping])

# 4. EVALUACIÓN Y VISUALIZACIÓN
# ---------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Precisión en Test: {test_acc*100:.2f}%")

# Gráficas de Aprendizaje
plt.figure(figsize=(12, 4))

# Gráfica de Loss (Error)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Curva de Pérdida (Loss)')
plt.xlabel('Época')
plt.legend()

# Gráfica de Accuracy (Precisión)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Curva de Precisión (Accuracy)')
plt.xlabel('Época')
plt.legend()

plt.tight_layout()
plt.show()

# 5. MATRIZ DE CONFUSIÓN (¿Qué ropa confunde con cuál?)
# ---------------------------------------------------------
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.title('Matriz de Confusión')
plt.show()