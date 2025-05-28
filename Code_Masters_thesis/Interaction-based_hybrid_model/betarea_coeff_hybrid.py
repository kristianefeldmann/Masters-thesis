
import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns


# ========================
# Section 0: Defining PyTorch
# ========================

# Setting random seed for PyTorch
torch_seed = 21
torch.manual_seed(torch_seed)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout = 0.1):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim)) # Linear layer
            layers.append(nn.ReLU()) # Activation function
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 16)
    
    def forward(self, x):
        x_hidden = self.hidden_layers(x)
        coeffs = self.output_layer(x_hidden)
        # Matching the input variable order above
        (   
            b0,             # intercept
            b_SiAl,         # Si/Al
            b_BET,          # BET
            b_micro,        # micropore
            b_mesovol,      # mesopore volume
            b_mesoarea,     # mesopore area
            b_C,            # NaOH concentration
            b_T,            # T
            b_t,            # time
            b_zeo,          # zeolite concentration
            b_Tt,           # T * t
            b_TC,           # T * C
            b_tC,            # t * C
            b_Tzeo,          # T * zeolite concentration
            b_tzeo,          # t * zeolite concentration
            b_Czeo          # C * zeolite concentration
        ) =     [coeffs[:, i] for i in range(16)]

        SiAl     = x[:, 0]
        BET      = x[:, 1]
        micropore = x[:, 2]
        mesovol  = x[:, 3]
        mesoarea = x[:, 4]
        C        = x[:, 5]
        T        = x[:, 6]
        t        = x[:, 7]
        zeoConc  = x[:, 8]

        y_pred = (
            b0
            + b_SiAl * SiAl
            + b_BET * BET
            + b_micro * micropore
            + b_mesovol * mesovol
            + b_mesoarea * mesoarea
            + b_C * C
            + b_T * T
            + b_t * t
            + b_zeo * zeoConc
            + b_Tt * T * t
            + b_TC * T * C
            + b_tC * t * C
            + b_Tzeo * T * zeoConc
            + b_tzeo * t * zeoConc
            + b_Czeo * C * zeoConc
        )

        return y_pred, coeffs

model = Net(
    input_dim = 13,
    hidden_dims = [192, 192, 224, 32, 192],
    dropout = 0.1
)

feature_columns = ['T/B0C_alktreat', 'Time/min_alktreat', 'NaOHConcentration/M_treat', 'Si/Al_prior', 'BETarea/m2g-1_prior', 'Microporevolume/cm3g-1_prior', 'Mesoporevolume/cm3g-1_prior', 'Mesoporearea/m2g-1_prior', 'zeoliteConcentration(g/mL)_alktreat']

# ========================
# Section 2: Set Up Reproducibility
# ========================
# Set environment variables for deterministic behavior
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Optional: Disable GPU for full determinism

# Set random seeds for Tensorflow
random_seed = 17  # Use the same seed every time
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# ========================
# Section 3: Load and Preprocess Data
# ========================
file_path = '/Users/Kristiane/Documents/5. klasse/H24/Prosjektoppgave/TensorFlow/final_noOut_med.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['BETarea_a/m2g-1_after'])
y = data['BETarea_a/m2g-1_after']

# Use a fixed random_state for reproducible splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# Section 4: Preparing x_tensor
# ========================

# Extract the columns from the test set
X_train_selected = X_train
X_test_selected = X_test

X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)
x_tensor = torch.tensor(X_test_selected_scaled, dtype = torch.float32)

# ========================
# Section 5: Training the PyTorch
# ========================

x_train_tensor = torch.tensor(X_train_selected_scaled, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32).view(-1, 1)

x_val_tensor = torch.tensor(X_test_selected_scaled, dtype = torch.float32)
y_val_tensor = torch.tensor(y_test.values, dtype = torch.float32).view(-1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

train_losses, val_losses = [], []

mean_target = y_train.mean()
min_delta = 0.01 * mean_target

class EarlyStopping:
    def __init__(self, patience=10, min_delta=min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter on meaningful improvement
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

# Instantiate early stopping
early_stopping = EarlyStopping(patience=10, min_delta=min_delta)

# Training loop with early stopping
train_losses, val_losses = [], []

for epoch in range(200):  # use a high max epoch, early stopping will break if needed
    model.train()
    optimizer.zero_grad()
    y_pred_train, _ = model(x_train_tensor)
    loss = criterion(y_pred_train.view(-1, 1), y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_val, _ = model(x_val_tensor)
        val_loss = criterion(y_pred_val.view(-1, 1), y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    print(f"Epoch {epoch}: Train loss = {loss.item():.6f}, val loss = {val_loss.item():.6f}")

    early_stopping(val_loss.item())
    if early_stopping.early_stop:
        print(f"Stopped early at epoch {epoch}")
        break

model.eval()

with torch.no_grad():
    y_pred, coeffs = model(x_tensor)
    print("Predictions:\n ", y_pred)
    print("Coefficients:\n", coeffs)

# ========================
# Section 6: Define the Model
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

    # Use a fixed optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae']
    )

    return model

# ========================
# Section 7: Train the Model
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
    validation_split=0.2,
    batch_size=16,  # Fixed batch size
    callbacks=[early_stopping],
    verbose=1
)

# ========================
# Section 8: Evaluate the Model
# ========================
test_loss, test_mae = final_model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test loss: {test_loss}, Test MAE: {test_mae}")

# Make predictions
predictions = final_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Normalize errors
mean_target = y_test.mean()
nmse = mse / (mean_target ** 2)
nmae = mae / mean_target

print(f"MSE: {mse}, MAE: {mae}")
print(f"Normalized MSE (NMSE): {nmse}")
print(f"Normalized MAE (NMAE): {nmae}")

# Compute Sum of Squares (SSQ)
ssq = np.sum((y_test.values - predictions.flatten()) ** 2)
ssq_net = np.sum((y_test.values - y_pred.numpy().flatten()) **2)
print(f"\nSum of Squares (SSQ) TensorFlow: {ssq:.6f}")
print(f"\nSum of Squares (SSQ) PyTorch: {ssq_net:.6f}")

mean_target = y_test.mean()
n_samples = len(y_test)

nssq_tf = ssq / (n_samples * mean_target**2)
nssq_pytorch = ssq_net / (n_samples * mean_target**2)

print(f"Normalized SSQ (TensorFlow): {nssq_tf:.6f}")
print(f"Normalized SSQ (PyTorch): {nssq_pytorch:.6f}")

# ========================
# Section 9: Model Summary
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

# ======================
# Section 10: Comparison of predictions
# ======================

y_pred_net = y_pred.numpy().flatten()
y_pred_tf = predictions.flatten()

df_compare = pd.DataFrame({
    'True_y': y_test.values,
    'PyTorch_Pred': y_pred_net,
    'TF_Pred': y_pred_tf
})

print(df_compare.head())


# Calculate 10% error margin for PyTorch predictions
true_values = y_test.values
pytorch_pred = y_pred.numpy().flatten()

error_margin = 0.1 * true_values
within_margin_pytorch = np.abs(pytorch_pred - true_values) <= error_margin
percentage_within_margin_pytorch = (np.sum(within_margin_pytorch) / len(true_values)) * 100

print(f"{np.sum(within_margin_pytorch)}/{len(true_values)} PyTorch predictions are within 10%. "
      f"({percentage_within_margin_pytorch:.2f}%)")


# Learning curves

# Normalize losses using mean target value
mean_train_target = y_train.mean()

# Normalize PyTorch losses
norm_train_losses = [loss / (mean_train_target**2) for loss in train_losses]
norm_val_losses = [loss / (mean_train_target**2) for loss in val_losses]

# Normalize TensorFlow losses (MAE → use mean directly instead of square)
norm_tf_train_losses = [loss / mean_train_target for loss in history.history['loss']]
norm_tf_val_losses = [loss / mean_train_target for loss in history.history['val_loss']]

plt.figure(figsize=(10, 6))
plt.plot(norm_tf_train_losses, label='TF Train Loss (Normalized MAE)', color='blue', linewidth=2)
plt.plot(norm_tf_val_losses, label='TF Val Loss (Normalized MAE)', color='skyblue', linewidth=2)
plt.plot(norm_train_losses, label='Hybrid Model Train Loss (Normalized MSE)', color='green', linestyle='--', linewidth=2)
plt.plot(norm_val_losses, label='Hybrid Model Val Loss (Normalized MSE)', color='lightgreen', linestyle='--', linewidth=2)

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Normalized Loss', fontsize=14)
plt.title('Normalized Learning Curves: TensorFlow vs Hybrid Model', fontsize=16)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



# Parity plot for both
plt.figure(figsize=(8, 6))

# Scatter points without transparency
plt.scatter(df_compare['True_y'], df_compare['PyTorch_Pred'], color='#7EC636', label='Coefficient predicting hybrid model', s=100)
plt.scatter(df_compare['True_y'], df_compare['TF_Pred'], color='#FF5D94', label='FFNN', s=100)

# Diagonal line (perfect prediction)
min_val, max_val = df_compare['True_y'].min(), df_compare['True_y'].max()
plt.plot([min_val, max_val], [min_val, max_val], color='#F59A23', linestyle='--')

# 10% error margin lines
plt.plot([min_val, max_val], [min_val * 0.9, max_val * 0.9], color='gray', linestyle=':', label='10% error margin')
plt.plot([min_val, max_val], [min_val * 1.1, max_val * 1.1], color='gray', linestyle=':')

# Calculate % within 10% margin
true = df_compare['True_y'].values
pytorch = df_compare['PyTorch_Pred'].values
tf = df_compare['TF_Pred'].values

within_margin_pt = np.abs(pytorch - true) <= 0.1 * true
within_margin_tf = np.abs(tf - true) <= 0.1 * true
perc_pt = (within_margin_pt.sum() / len(true)) * 100
perc_tf = (within_margin_tf.sum() / len(true)) * 100

# Textbox content
textstr = (
    f"Coefficient predicting hybrid model:\n"
    f"SSQ = {ssq_net:.4f}\n"
    f"{within_margin_pt.sum()}/{len(df_compare)} ({perc_pt:.2f}%) within 10%\n\n"
    f"FFNN:\n"
    f"SSQ = {ssq:.4f}\n"
    f"{within_margin_tf.sum()}/{len(df_compare)} ({perc_tf:.2f}%) within 10%"
)
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray')

# Place textbox in bottom right
plt.text(max_val, min_val, textstr, fontsize=11, ha='right', va='bottom', bbox=props)

# Labels and formatting
plt.xlabel('Actual BET area [m2/g]', fontsize=14)
plt.ylabel('Predicted BET area [m2/g]', fontsize=14)
plt.title('Parity plot: Coefficient predicting hybrid model vs FFNN', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

print(min_delta)
print(f"PyTorch model trained for {len(train_losses)} epochs")

# === PyTorch Metrics ===
mse_pytorch = mean_squared_error(y_test, y_pred_net)
mae_pytorch = mean_absolute_error(y_test, y_pred_net)

nmse_pytorch = mse_pytorch / (mean_target ** 2)
nmae_pytorch = mae_pytorch / mean_target

print(f"PyTorch MSE: {mse_pytorch}, MAE: {mae_pytorch}")
print(f"PyTorch Normalized MSE (NMSE): {nmse_pytorch}")
print(f"PyTorch Normalized MAE (NMAE): {nmae_pytorch}")

# Extract the last set of coefficients
last_coeffs = coeffs[-1]

# Define the coefficient names and corresponding variable names
coeff_names = [
    "Intercept (b0)", "Si/Al (b_SiAl)", "BET (b_BET)", "Micropore volume (b_micro)",
    "Mesopore volume (b_mesovol)", "Mesopore area (b_mesoarea)", "NaOH concentration (b_C)",
    "Temperature (b_T)", "Time (b_t)", "Zeolite concentration (b_zeo)",
    "T * t (b_Tt)", "T * C (b_TC)", "t * C (b_tC)",
    "T * Zeolite concentration (b_Tzeo)", "t * Zeolite concentration (b_tzeo)", "C * Zeolite concentration (b_Czeo)"
]

# Print the last set of coefficients
print("\nLast set of coefficients:\n")
for name, value in zip(coeff_names, last_coeffs):
    print(f"{name}: {value.item():.4f}")