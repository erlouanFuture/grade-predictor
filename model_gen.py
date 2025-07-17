import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Define the model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # For binary classification
])

# 2. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Train the model with dummy data
import numpy as np
X_train = np.random.random((100, 10))
y_train = np.random.randint(2, size=(100, 1))

model.fit(X_train, y_train, epochs=5, batch_size=10)

# 4. Save the model in .keras format
model.save("my_model.keras")
