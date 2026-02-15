import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import kagglehub
import os

# 1. DESCARGA Y CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------
path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

# Estructura carpetas: path + 'horse-or-human' + 'train/validation'
train_dir = os.path.join(path, 'horse-or-human', 'train')
validation_dir = os.path.join(path, 'horse-or-human', 'validation')

print(f"Ruta de entrenamiento: {train_dir}")
print(f"Ruta de validación: {validation_dir}")

# 2. GENERADORES DE DATOS
# -----------------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1.0/255.)
val_datagen  = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64), 
    batch_size=32,
    class_mode='binary' # Requisito del ejercicio
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 3. MODELO LeNet-5 (Adaptado para binario)
# -----------------------------------------------------------
# Seguimos la arquitectura de referencia LeNet-5
model = models.Sequential([
    # Capa 1: 6 filtros + Pooling
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(64, 64, 3)),
    layers.AveragePooling2D(),

    # Capa 2: 16 filtros + Pooling
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.AveragePooling2D(),

    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    
    # SALIDA BINARIA: 1 sola unidad con Sigmoid
    layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', # Crucial para el ejercicio
              metrics=['accuracy'])

# 4. ENTRENAMIENTO
# -----------------------------------------------------------
print("\n¡Iniciando el entrenamiento! 🚀")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 5. GRÁFICAS (Visualización de desempeño)
# -----------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión (Accuracy)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (Loss)')
plt.legend()

plt.show()