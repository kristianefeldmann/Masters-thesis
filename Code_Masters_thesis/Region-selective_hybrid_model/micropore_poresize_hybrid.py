import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
random_seed = 13
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# ========================
# Section 2: Load and preprocess data
# ========================

file_path = '/Users/Kristiane/Documents/5. klasse/H24/Prosjektoppgave/TensorFlow/final_noOut_med.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['Microporevolume/cm3g-1_after'])
y = data['Microporevolume/cm3g-1_after']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# ========================
# Section 3: Hybrid Model Class Definition
# ========================

class HybridModel(tf.keras.Model):
    def __init__(self, layer_sizes, lambda_reg=0.01):
        super(HybridModel, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(size, activation='swish') for size in layer_sizes]
        self.output_layer = tf.keras.layers.Dense(1)

        # Domain-informed initial values (as variables so they can be trained)
        self.weight_low = tf.Variable(-0.2, trainable=True, dtype=tf.float32)
        self.weight_opt = tf.Variable(0.1, trainable=True, dtype=tf.float32)
        self.weight_high = tf.Variable(-0.1, trainable=True, dtype=tf.float32)

        self.low_thresh = 25.0
        self.high_thresh = 50.0
        self.severity_baseline = 390.0
        self.lambda_reg = lambda_reg

    def call(self, x):
        si_al = x[:, 0]
        naoh = x[:, 5]
        time = x[:, 7]
        temp = x[:, 6]

        severity = naoh * time * temp
        severity_norm = severity / self.severity_baseline

        low_thresh_scaled = self.low_thresh / severity_norm
        high_thresh_scaled = self.high_thresh / severity_norm

        hybrid_signal = tf.where(
            si_al < low_thresh_scaled, self.weight_low,
            tf.where(si_al <= high_thresh_scaled, self.weight_opt, self.weight_high)
        )
        hybrid_signal = tf.expand_dims(hybrid_signal, axis=-1)

        out = x
        for layer in self.hidden_layers:
            out = layer(out)

        out = tf.concat([out, hybrid_signal], axis=1)
        prediction = self.output_layer(out)
        return prediction, hybrid_signal

# ========================
# Section 4: Train the Model
# ========================

model = HybridModel([192, 192, 224, 32, 192], lambda_reg=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
mae_loss = tf.keras.losses.MeanAbsoluteError()

mean_target = y_train.mean()
min_delta = 0.01 * mean_target

class EarlyStopping:
    def __init__(self, patience = 10, min_delta = min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(16).shuffle(100)
val_tensor = tf.convert_to_tensor(X_val_scaled, dtype = tf.float32)
y_val_tensor = tf.convert_to_tensor(y_val.values, dtype = tf.float32)

epochs = 200
early_stopping = EarlyStopping(patience = 10, min_delta = 0.01 * y_train.mean())
history = {"train_loss": [], "val_loss": []}

for epoch in range (epochs):
    epoch_loss = 0

    for batch_x, batch_y in train_dataset:
        with tf.GradientTape() as tape:
            y_pred, hybrid_signal = model(batch_x)
            base_loss = mae_loss(batch_y, tf.squeeze(y_pred))
            total_loss = base_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss += total_loss.numpy()

    history["train_loss"].append(epoch_loss)

    y_val_pred, _ = model(val_tensor)
    val_loss = mae_loss(y_val_tensor, tf.squeeze(y_val_pred)).numpy()
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch+1}: Train loss = {epoch_loss:.4f}, val loss = {val_loss:.4f}")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Stopped early at epoch {epoch+1}")
        break

# ========================
# Section 5: Evaluate on Test Set
# ========================
y_pred_test, _ = model(X_test_scaled)
test_mae = mae_loss(y_test, tf.squeeze(y_pred_test)).numpy()
mse = mean_squared_error(y_test, y_pred_test.numpy())
mae = mean_absolute_error(y_test, y_pred_test.numpy())

mean_target = y_test.mean()
nmse = mse / (mean_target ** 2)
nmae = mae / mean_target

print(f"Test MAE: {test_mae:.4f}")
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, NMSE: {nmse:.4f}, NMAE: {nmae:.4f}")


# ========================
# Section 6: Parity Plot
# ========================
true_values = y_test.values
predicted_values = y_pred_test.numpy().flatten()
error_margin = 0.1 * true_values
within_margin = abs(predicted_values - true_values) <= error_margin
percentage_within_margin = (sum(within_margin) / len(true_values)) * 100

print(f"{sum(within_margin)}/{len(true_values)} predictions are within 10%. ({percentage_within_margin:.2f}%)")

plt.figure(figsize=(8, 6))
plt.scatter(true_values, predicted_values, color='#F59A23', label='Predictions', s=100)
plt.scatter(true_values[within_margin], predicted_values[within_margin], color='#7EC636', label='Within 10% error margin', s=100)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linewidth=2)
plt.legend(fontsize=12, loc='upper left')
plt.xlabel('Actual BET area [m2/g]', fontsize=14)
plt.ylabel('Predicted BET area [m2/g]', fontsize=14)
plt.title('Parity Plot', fontsize=16)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ========================
# Section 7: Plot Training Loss
# ========================
plt.figure(figsize=(10, 6))

plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss (MAE)', fontsize=14)
plt.title('Training Loss Curve', fontsize=16)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
