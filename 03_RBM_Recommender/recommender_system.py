import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import kagglehub
import os

# ==========================================
# 1. DESCARGA Y LECTURA MANUAL (A prueba de fallos)
# ==========================================
print("Descargando dataset... ⏳")
# Esto solo baja los archivos a tu disco, no intenta leerlos (así no falla)
path = kagglehub.dataset_download("zygmunt/goodbooks-10k")
print(f"Dataset descargado en: {path}")

# Construimos las rutas manuales
ratings_file = os.path.join(path, "ratings.csv")
books_file = os.path.join(path, "books.csv")

print("Leyendo archivos con Pandas (Modo tolerante)... 🐼")

# LEEMOS MANUALMENTE CON PANDAS
# encoding='latin-1': Para que no explote con tildes raras.
# on_bad_lines='skip': La clave mágica. Si una línea está rota, la salta.
ratings = pd.read_csv(ratings_file, encoding="latin-1", on_bad_lines='skip')
books = pd.read_csv(books_file, encoding="latin-1", on_bad_lines='skip')

print(f"✅ ¡LOGRADO! Datos cargados: {len(ratings)} ratings y {len(books)} libros.")

# ==========================================
# 2. PREPROCESAMIENTO
# ==========================================
# Filtramos top 1000 para no reventar la RAM
top_books_ids = ratings.groupby('book_id')['rating'].count().sort_values(ascending=False).head(1000).index
filtered_ratings = ratings[ratings['book_id'].isin(top_books_ids)]

# Matriz Usuario-Libro
user_book_matrix = filtered_ratings.pivot_table(index='user_id', columns='book_id', values='rating')
user_book_matrix.fillna(0, inplace=True)

# Normalizar (0 a 1)
X = user_book_matrix.values / 5.0

# Split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

num_books = X.shape[1]
print(f"Matriz lista: {X.shape[0]} usuarios x {num_books} libros")

# ==========================================
# 3. MODELO (Autoencoder)
# ==========================================
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(num_books,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'), # Cuello de botella
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_books, activation='sigmoid') # Salida 0-1
])

model.compile(optimizer='adam', loss='mean_squared_error')

# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
print("\nIniciando entrenamiento... 🚀")
history = model.fit(
    X_train, X_train,
    epochs=20, # Puedes bajarlo a 10 si tienes prisa
    batch_size=64,
    validation_data=(X_test, X_test),
    verbose=1
)

# ==========================================
# 5. RESULTADOS
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Error de Reconstrucción')
plt.legend()
plt.show()

# --- RECOMENDACIÓN FINAL ---
print("\n🔎 RECOMENDACIONES PARA EL USUARIO #10:")
user_idx = 10
user_data = X_test[user_idx].reshape(1, -1)
preds = model.predict(user_data) * 5
real = user_data[0] * 5

print("\n📚 LE GUSTAN (Real):")
liked_idx = np.where(real >= 4)[0][:5]
for idx in liked_idx:
    bid = user_book_matrix.columns[idx]
    try:
        title = books.loc[books['book_id'] == bid, 'original_title'].values[0]
        print(f" - {title}")
    except:
        print(f" - Libro ID {bid} (Título no encontrado)")

print("\n🤖 LA IA RECOMIENDA:")
# Libros no leídos (0) con predicción alta
rec_idx = np.where((real == 0) & (preds > 3.5))[0]
rec_idx = sorted(rec_idx, key=lambda x: preds[x], reverse=True)[:5]

for idx in rec_idx:
    bid = user_book_matrix.columns[idx]
    try:
        title = books.loc[books['book_id'] == bid, 'original_title'].values[0]
        score = preds[idx]
        print(f" - {title} (Predicción: {score:.1f}/5)")
    except:
        print(f" - Libro ID {bid} (Predicción: {preds[idx]:.1f}/5)")