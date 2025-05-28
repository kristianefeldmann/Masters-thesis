import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========================
# Section 1: Set up reproducibility
# ========================

# Set environment variables for reproducible behaviour
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set random seeds
random_seed = 12
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# ========================
# Section 2: Load and preprocess data
# ========================

file_path = '/Users/Kristiane/Documents/5. klasse/H24/Prosjektoppgave/TensorFlow/final_noOut_med.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['BETarea_a/m2g-1_after'])
y = data['BETarea_a/m2g-1_after']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# Section 3: Define the model
# ========================

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(192, activation='swish'),
        tf.keras.layers.Dense(192, activation='swish'),
        tf.keras.layers.Dense(224, activation='swish'),
        tf.keras.layers.Dense(32, activation='swish'),
        tf.keras.layers.Dense(192, activation='swish'),
        tf.keras.layers.Dense(1)  # Output layer
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae']
    )

    return model

# ========================
# Section 4: Train the model
# ========================

# Clear any existing TensorFlow session
tf.keras.backend.clear_session()

# Build the model
final_model = build_model()

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = final_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size = 16,  # Fixed batch size
    verbose=1
)

# ========================
# Section 5: Evaluate the model
# ========================
test_loss, test_mae = final_model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test loss: {test_loss}, Test MAE: {test_mae}")

predictions = final_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Normalize errors
mean_target = y_test.mean()
nmse = mse / (mean_target ** 2)
nmae = mae / mean_target

print(f"MSE: {mse}, MAE: {mae}")
print(f"Normalized MSE (NMSE): {nmse}")
print(f"Normalized MAE (NMAE): {nmae}")

# ========================
# Section 6: Parity Plot
# ========================

true_values = y_test.values
predicted_values = predictions.flatten()
error_margin = 0.1 * true_values
within_margin = abs(predicted_values - true_values) <= error_margin
percentage_within_margin = (sum(within_margin) / len(true_values)) * 100
print(f"{sum(within_margin)}/{len(true_values)} predictions are within 10%. ({percentage_within_margin:.2f}%)")

plt.figure(figsize=(8, 6))
plt.scatter(true_values, predicted_values, color='#F59A23', label='Predictions', s=100)
plt.scatter(
    true_values[within_margin],
    predicted_values[within_margin],
    color='#7EC636',
    label='Within 10% error margin',
    s=100
)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linewidth=2)
plt.legend(fontsize=12, loc='upper left')
plt.xlabel('Actual micropore volume [cm3/g]', fontsize=14)
plt.ylabel('Predicted micropore volume [cm3/g]', fontsize=14)
plt.title('Parity Plot', fontsize=16)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ========================
# Section 7: Plot training and validation loss
# ========================

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss (MAE)', fontsize=14)
plt.title('Training and Validation Loss', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ========================
# Section 8: Model summary
# ========================
def print_model_summary(model):
    print("\n===== Layer-wise Parameter Distribution =====\n")
    
    total_params = model.count_params()
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_params = layer.count_params()
        print(f"Layer {i+1}: {layer_name} | Parameters: {layer_params}")

    print(f"\n **Total Trainable Parameters:** {total_params}\n")

print_model_summary(final_model)