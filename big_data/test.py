import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

btc_file = '../knn/Bitcoin_01.10.2017-20.03.2025_historical_data_coinmarketcap.csv'
eth_file = '../knn/Ethereum_01.12.2018-28.02.2025_historical_data_coinmarketcap.csv'

df_btc = pd.read_csv(btc_file, sep=';')
df_eth = pd.read_csv(eth_file, sep=';')

scaler = MinMaxScaler()
scaled_df = df_btc.copy()
scaled_df[['open', 'high', 'low', 'close', 'volume', 'marketCap']] = scaler.fit_transform(
    scaled_df[['open', 'high', 'low', 'close', 'volume', 'marketCap']]
)

scaled_df.drop(columns=['timeOpen', 'timeClose', 'timeHigh', 'timeLow'], inplace=True)
scaled_df = scaled_df.set_index('timestamp')

print(scaled_df.head().dtypes)

def prepare_data(df, window_size=5):
    X, y = [], []
    for i in range(len(df) - window_size):
        features = df.iloc[i:i+window_size].drop(columns=['close']).values
        target = df.iloc[i+window_size]['close']
        X.append(features.flatten())
        y.append(target)
    return np.array(X), np.array(y)


import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.model(x)


import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_torch_model(model_name, model, X_train, y_train, X_test, y_test, scaler, results_dict):
    print(f"Training {model_name}...")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    preds_inv = scaler.inverse_transform(preds)
    y_true_inv = scaler.inverse_transform(y_true)
    rmse = mean_squared_error(y_true_inv, preds_inv)
    results_dict[model_name] = rmse

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv, label='Actual')
    plt.plot(preds_inv, label='Predicted')
    plt.title(f"{model_name} Prediction (RMSE: {rmse:.4f})")
    plt.legend()
    plt.grid(True)
    plt.show()


results = {}

for i in range(5, 40, 5):
    X, y = prepare_data(scaled_df, window_size=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    
    input_size = X_train.shape[1]
    hidden_size = min(128, 2 * i)
    model = MLPModel(input_size=input_size, hidden_size=hidden_size)

    run_torch_model(f"MLP_{i}", model, X_train, y_train, X_test, y_test, scaler, results)

print("\n📈 RMSE Results Summary:")
for model_name, rmse in results.items():
    print(f"{model_name:20s}: RMSE = {rmse:.4f}")
