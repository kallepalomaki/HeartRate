import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic multi-wavelength time series data
n_samples = 1000
seq_len = 100  # length of each time series
channels = 3   # green, red, IR

# Simulate signals with sine waves and noise
def simulate_ppg_waveform(freq, noise=0.05):
    t = np.linspace(0, 1, seq_len)
    return np.sin(2 * np.pi * freq * t) + noise * np.random.randn(seq_len)

X = np.zeros((n_samples, channels, seq_len))
y = np.zeros(n_samples)

np.random.seed(42)
for i in range(n_samples):
    mean_hr = 75  # mean resting HR in bpm
    std_dev_hr = 10  # standard deviation in bpm
    hr = np.random.normal(mean_hr, std_dev_hr)  # generate HR with normal distribution
    freq = hr / 60  # convert to Hz
    X[i, 0, :] = simulate_ppg_waveform(freq, noise=0.1)  # green
    X[i, 1, :] = simulate_ppg_waveform(freq, noise=0.12)  # red
    X[i, 2, :] = simulate_ppg_waveform(freq, noise=0.15)  # IR
    y[i] = hr

# Train/test split
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

# 2. Define CNN model
class PPGCNN(nn.Module):
    def __init__(self):
        super(PPGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))      # [B, 16, 100]
        x = self.pool(x)                  # [B, 16, 1]
        x = x.view(x.size(0), -1)         # [B, 16]
        x = self.fc(x)                    # [B, 1]
        return x

# 3. Train the model
model = PPGCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.2f}")

# 4. Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).squeeze().numpy()
    targets = y_test_tensor.squeeze().numpy()
    mae = np.mean(np.abs(preds - targets))
    print(f"\nTest MAE: {mae:.2f} bpm")

# 5. Plot predictions
plt.figure(figsize=(10, 4))
plt.plot(targets[:100], label="True HR")
plt.plot(preds[:100], label="Predicted HR", linestyle='--')
plt.title("Heart Rate Prediction from Multi-Wavelength PPG (CNN)")
plt.xlabel("Sample Index")
plt.ylabel("Heart Rate (bpm)")
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()  # Clear the figure to avoid any interference with the next plot

# 6. Plot PPG data
plt.figure(figsize=(10, 4))
plt.plot(X[0,0,:], label="PPG Signal")  # Assuming X is your input PPG data
plt.title("PPG Signal for Sample 0")
plt.xlabel("Sample Index")
plt.ylabel("PPG Amplitude")
plt.legend()
plt.tight_layout()
plt.show()  # Display the second plot
# 5. Plot predictions
#plt.figure(figsize=(10, 4))
#plt.plot(X[0,0,:], label="ppg")

#plt.legend()
#plt.tight_layout()
#plt.show()