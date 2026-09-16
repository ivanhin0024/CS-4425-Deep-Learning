
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

torch.manual_seed(0)

# Part 1

housing = fetch_california_housing()

print(housing.DESCR) #On my machine, DESCR is underlined with red but it runs fine. I think it is a bug in PyCharm.

df = pd.DataFrame(housing.data, columns=housing.feature_names) # same with data and feature_names but still runs and works 

df["MedHouseVal"] = housing.target # same with target but still runs and works 

print("==== df.head() ====")
print(df.head())

print()
print("==== df.describe() ====")
print(df.describe())

# Part 2

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Part 3 Model 1: Baseline linear regression

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Part 3 Model 2: Neural network

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the network
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



model = MLP(input_dim=X_train_tensor.shape[1], hidden_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


EPOCHS = 100
loss_history = []

print()
print("--- Training the neural network ---")

for epoch in range(1, EPOCHS + 1):

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())


    if epoch % 10 == 0:
        print(f"Epoch [{epoch:3d}/{EPOCHS}]  Loss (MSE): {loss.item():.6f}")

# Part 4

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Model 2 (PyTorch MLP): Training Loss vs. Epoch")
plt.grid(alpha=0.3)
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")

print()
print("Saved plot to loss_curve.png")