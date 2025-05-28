import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# ========================
# Section 1: Set up reproducibility
# ========================

# Set environment variables for reproducible behaviour
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set random seeds
random_seed = 35
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# ========================
# Section 2: Load and preprocess data
# ========================

file_path = '/Users/Kristiane/Documents/5. klasse/H24/Prosjektoppgave/TensorFlow/final_noOut_med.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['Mesoporearea/m2g-1_after'])
y = data['Mesoporearea/m2g-1_after']

# ========================
# Section 3: Making linear regression model from literatur data for hybrid model
# ========================

# Setting random seed for PyTorch
torch_seed = 12
torch.manual_seed(torch_seed)

# Input of literature data
si_al_lit = np.array([596, 139, 64, 56]).reshape(-1, 1)
a_lit = np.array([1.90e-4, 7.10-5, 6.00e-5, 1.80e-4])
p_lit = np.array([3.137e-6, 3.175e-7, 2.329e-6, 1.735e-6])
m_lit = np.array([1.77, 2.08, 1.75, 2.00])
kd_lit = np.array([1.277e-5, 9.056e-6, 8.720e-6, 8.396e-6])

# Fitting linear modls
poly = PolynomialFeatures(degree = 2)
si_al_poly = poly.fit_transform(si_al_lit)

model_a = LinearRegression().fit(si_al_poly, a_lit)
model_p = LinearRegression().fit(si_al_poly, p_lit)
model_m = LinearRegression().fit(si_al_poly, m_lit)
model_kd = LinearRegression().fit(si_al_poly, kd_lit)

si_al_range = np.linspace(50, 600, 200).reshape(-1, 1)
si_al_range_poly = poly.transform(si_al_range)

a_pred = model_a.predict(si_al_range_poly)
p_pred = model_p.predict(si_al_range_poly)
m_pred = model_m.predict(si_al_range_poly)
kd_pred = model_kd.predict(si_al_range_poly)

# Standard data
si_al_standard = data["Si/Al_prior"].values.reshape(-1, 1)
si_al_standard_poly = poly.transform(si_al_standard)

a_user_pred = model_a.predict(si_al_standard_poly)
p_user_pred = model_p.predict(si_al_standard_poly)
m_user_pred = model_m.predict(si_al_standard_poly)
kd_user_pred = model_kd.predict(si_al_standard_poly)

# Saving the predicted values
predicted_params = pd.DataFrame({
    "Si/Al_prior": si_al_standard.flatten(),
    "a_pred": a_user_pred,
    "p_pred": p_user_pred,
    "m_pred": m_user_pred,
    "kd_pred": kd_user_pred
})

"""
# Plotting polynomial regression models with both training data and predicted standard data

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# Plot for a
axs[0].scatter(si_al_lit, a_lit, color='red', label='Literature Data')
axs[0].plot(si_al_range, a_pred, label='Polynomial Fit')
axs[0].scatter(si_al_standard, a_user_pred, color='green', marker='x', s=100, label='Standard Data Prediction')
axs[0].set_title('a vs Si/Al')
axs[0].set_xlabel('Si/Al')
axs[0].set_ylabel('a')
axs[0].legend()

# Plot for p
axs[1].scatter(si_al_lit, p_lit, color='red', label='Literature Data')
axs[1].plot(si_al_range, p_pred, label='Polynomial Fit')
axs[1].scatter(si_al_standard, p_user_pred, color='green', marker='x', s=100, label='Standard Data Prediction')
axs[1].set_title('p vs Si/Al')
axs[1].set_xlabel('Si/Al')
axs[1].set_ylabel('p')
axs[1].legend()

# Plot for m
axs[2].scatter(si_al_lit, m_lit, color='red', label='Literature Data')
axs[2].plot(si_al_range, m_pred, label='Polynomial Fit')
axs[2].scatter(si_al_standard, m_user_pred, color='green', marker='x', s=100, label='Standard Data Prediction')
axs[2].set_title('m vs Si/Al')
axs[2].set_xlabel('Si/Al')
axs[2].set_ylabel('m')
axs[2].legend()

# Plot for kd
axs[3].scatter(si_al_lit, kd_lit, color='red', label='Literature Data')
axs[3].plot(si_al_range, kd_pred, label='Polynomial Fit')
axs[3].scatter(si_al_standard, kd_user_pred, color='green', marker='x', s=100, label='Standard Data Prediction')
axs[3].set_title('kd vs Si/Al')
axs[3].set_xlabel('Si/Al')
axs[3].set_ylabel('kd')
axs[3].legend()

plt.tight_layout()
plt.show()
"""


# Calculating C_Si(t)
t = data["Time/min_alktreat"].values
A_BET = data["BETarea/m2g-1_prior"].values
data["m0"] = data["zeoliteConcentration(g/mL)_alktreat"] * 1000  # [g/dm³]
m0 = data["m0"].values
Si0 = si_al_standard.flatten()

a_dyn = a_user_pred.flatten()
p_dyn = p_user_pred.flatten()
m_dyn = m_user_pred.flatten()
kd_dyn = kd_user_pred.flatten()

# ========================
# Section 4: Defining hybrid model
# ========================

class HybridNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        self.hidden_layers = nn.Sequential(*layers)

        self.fc_extract_physical = nn.Linear(in_dim, 3) # Si/Al, BET area and zeolite concentration

        # Output layer receives one extra feature, C_Si
        self.output_layer = nn.Linear(in_dim + 1, 1)

    def forward(self, x):
        x_hidden = self.hidden_layers(x)

        # For equation using the output variables
        physical_preds = self.fc_extract_physical(x_hidden)
        si_al_pred = physical_preds[:, 0]
        BET_pred = physical_preds[:, 1]
        zeoConc_pred = physical_preds[:, 2]
        t_input = x[:, 7]

        si_al_std = si_al_pred.detach().cpu().numpy().reshape(-1, 1)
        si_al_trans_poly = poly.transform(si_al_std)
        kd_user_trans_poly = torch.tensor(
            model_kd.predict(si_al_trans_poly),
            dtype = x.dtype,
            device = x.device
        )

        A = zeoConc_pred * BET_pred
        beta = kd_user_trans_poly / si_al_pred
        C_si_t = beta * A * si_al_pred * t_input
        C_si_t = C_si_t.unsqueeze(1)

        x_concat = torch.cat((x_hidden, C_si_t), dim = 1)
        y_pred = self.output_layer(x_concat)

        return y_pred


model = HybridNet(
    input_dim = 13,
    hidden_dims = [192, 192, 224, 32, 192],
    dropout = 0.1
)

# ======================
# Section 5: Defining training and test sets
# ======================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



with torch.no_grad():
    x_all_tensor = torch.tensor(scaler.transform(X), dtype=torch.float32)
    x_hidden = model.hidden_layers(x_all_tensor)
    physical_preds = model.fc_extract_physical(x_hidden).numpy()

# True physical values
si_al_true = X["Si/Al_prior"].values
bet_true = X["BETarea/m2g-1_prior"].values
zeoconc_true = X["zeoliteConcentration(g/mL)_alktreat"].values * 1000  # convert to g/dm³

# Plot correlation heatmap
labels = ["Latent_0", "Latent_1", "Latent_2"]
true_labels = ["Si/Al", "BET", "zeoConc"]
all_corrs = []

for i, true_values in enumerate([si_al_true, bet_true, zeoconc_true]):
    corrs = []
    for j in range(3):
        r, _ = pearsonr(physical_preds[:, j], true_values)
        corrs.append(r)
    all_corrs.append(corrs)

corr_df = pd.DataFrame(all_corrs, index=true_labels, columns=labels)

# Plot heatmap
#plt.figure(figsize=(6, 4))
#sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
#plt.title("Correlation between physical_preds and true variables")
#plt.tight_layout()
#plt.show()

# ======================
# Section 6: Training the hybrid model
# ======================

x_train_tensor = torch.tensor(X_train_scaled, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32).view(-1, 1)

x_val_tensor = torch.tensor(X_test_scaled, dtype = torch.float32)
y_val_tensor = torch.tensor(y_test.values, dtype = torch.float32).view(-1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

train_losses, val_losses = [], []

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

early_stopping = EarlyStopping(patience = 10, min_delta = min_delta)


# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(x_train_tensor)
    loss = criterion(y_pred_train.view(-1, 1), y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_val = model(x_val_tensor)
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
    y_pred = model(x_val_tensor)
    print("Predictions:\n ", y_pred)

# ========================
# Section 7: Define the model
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
# Section 8: Train the model
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
    batch_size = 16,  # Fixed batch size
    callbacks=[early_stopping],
    verbose=1
)

# ========================
# Section 9: Evaluate the Model
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
# Section 10: Model Summary
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
# Section 11: Comparison of predictions
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
plt.xlabel('Actual mesopore area [m2/g]', fontsize=14)
plt.ylabel('Predicted mesopore area [m2/g]', fontsize=14)
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