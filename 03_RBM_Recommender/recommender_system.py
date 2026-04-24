import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import kagglehub
import os

# ==========================================
# 1. DOWNLOAD AND MANUAL READING (Fault-tolerant)
# ==========================================
print("Downloading dataset... ⏳")
# Downloads the files to local disk.
path = kagglehub.dataset_download("zygmunt/goodbooks-10k")
print(f"Dataset downloaded at: {path}")

# Build file paths manually
ratings_file = os.path.join(path, "ratings.csv")
books_file = os.path.join(path, "books.csv")

print("Reading files with Pandas (Tolerant mode)... 🐼")

# READ MANUALLY WITH PANDAS
# encoding='latin-1': prevents failures on special characters/accents.
# on_bad_lines='skip': skips broken lines instead of crashing.
ratings = pd.read_csv(ratings_file, encoding="latin-1", on_bad_lines='skip')
books = pd.read_csv(books_file, encoding="latin-1", on_bad_lines='skip')

print(f"✅ SUCCESS! Data loaded: {len(ratings)} ratings and {len(books)} books.")

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Filter top 1000 books to avoid running out of RAM
top_books_ids = ratings.groupby('book_id')['rating'].count().sort_values(ascending=False).head(1000).index
filtered_ratings = ratings[ratings['book_id'].isin(top_books_ids)]

# User-Book matrix
user_book_matrix = filtered_ratings.pivot_table(index='user_id', columns='book_id', values='rating')
user_book_matrix.fillna(0, inplace=True)

# Normalize ratings to 0-1 range
X = user_book_matrix.values / 5.0

# Train/test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

num_books = X.shape[1]
print(f"Matrix ready: {X.shape[0]} users x {num_books} books")

# ==========================================
# 3. MODEL (Autoencoder)
# ==========================================
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(num_books,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'), # Bottleneck
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_books, activation='sigmoid') # Output in 0-1 range
])

model.compile(optimizer='adam', loss='mean_squared_error')

# ==========================================
# 4. TRAINING
# ==========================================
print("\nStarting training... 🚀")
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, X_test),
    verbose=1
)

# ==========================================
# 5. RESULTS
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Reconstruction Error')
plt.legend()
plt.show()

# --- FINAL RECOMMENDATION ---
print("\n🔎 RECOMMENDATIONS FOR USER #10:")
user_idx = 10
user_data = X_test[user_idx].reshape(1, -1)
preds = model.predict(user_data) * 5
real = user_data[0] * 5

print("\n📚 BOOKS THEY LIKED (Actual ratings):")
liked_idx = np.where(real >= 4)[0][:5]
for idx in liked_idx:
    bid = user_book_matrix.columns[idx]
    try:
        title = books.loc[books['book_id'] == bid, 'original_title'].values[0]
        print(f" - {title}")
    except:
        print(f" - Book ID {bid} (Title not found)")

print("\n🤖 AI RECOMMENDATIONS:")
# Unread books (0) with a high predicted score
rec_idx = np.where((real == 0) & (preds > 3.5))[0]
rec_idx = sorted(rec_idx, key=lambda x: preds[x], reverse=True)[:5]

for idx in rec_idx:
    bid = user_book_matrix.columns[idx]
    try:
        title = books.loc[books['book_id'] == bid, 'original_title'].values[0]
        score = preds[idx]
        print(f" - {title} (Predicted: {score:.1f}/5)")
    except:
        print(f" - Book ID {bid} (Predicted: {preds[idx]:.1f}/5)")
