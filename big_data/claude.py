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

# Завантаження даних
btc_file = '../knn/Bitcoin_01.10.2017-20.03.2025_historical_data_coinmarketcap.csv'
eth_file = '../knn/Ethereum_01.12.2018-28.02.2025_historical_data_coinmarketcap.csv'

df_btc = pd.read_csv(btc_file, sep=';')
df_eth = pd.read_csv(eth_file, sep=';')

# Підготовка даних з окремими скейлерами
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_df = df_btc.copy()

# Видаляємо непotрібні колонки
columns_to_drop = ['timeOpen', 'timeClose', 'timeHigh', 'timeLow']
for col in columns_to_drop:
    if col in scaled_df.columns:
        scaled_df.drop(columns=[col], inplace=True)

# Встановлюємо timestamp як індекс
if 'timestamp' in scaled_df.columns:
    scaled_df = scaled_df.set_index('timestamp')

# Масштабування features (все окрім close)
feature_columns = ['open', 'high', 'low', 'volume', 'marketCap']
target_column = 'close'

# Перевіряємо, чи всі колонки існують
available_feature_cols = [col for col in feature_columns if col in scaled_df.columns]
print(f"Доступні колонки для features: {available_feature_cols}")

if target_column not in scaled_df.columns:
    print(f"Помилка: колонка '{target_column}' не знайдена!")
    exit()

# Масштабування
scaled_df[available_feature_cols] = feature_scaler.fit_transform(scaled_df[available_feature_cols])
scaled_df[[target_column]] = target_scaler.fit_transform(scaled_df[[target_column]])

print("Структура даних:")
print(scaled_df.head())
print(f"Shape: {scaled_df.shape}")

def prepare_data(df, window_size=5):
    """
    Підготовка даних для навчання з вікном часу
    """
    X, y = [], []
    feature_cols = [col for col in df.columns if col != 'close']
    
    for i in range(len(df) - window_size):
        # Беремо features за window_size днів
        features = df[feature_cols].iloc[i:i+window_size].values
        # Target - ціна закриття наступного дня
        target = df['close'].iloc[i+window_size]
        
        X.append(features.flatten())
        y.append(target)
    
    return np.array(X), np.array(y)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2=None, dropout_rate=0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        if hidden_size2:
            layers.append(nn.Linear(hidden_size1, hidden_size2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_size2, 1))
        else:
            layers.append(nn.Linear(hidden_size1, 1))
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def run_torch_model(model_name, model, X_train, y_train, X_test, y_test, 
                   target_scaler, results_dict, epochs=100, lr=0.001):
    """
    Навчання та оцінка PyTorch моделі
    """
    print(f"Training {model_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Конвертація в тензори
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Навчання
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Прогнозування
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()
    
    # Зворотнє масштабування (тільки для target)
    preds_inv = target_scaler.inverse_transform(preds)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Метрики
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    mae = mean_absolute_error(y_test_inv, preds_inv)
    
    results_dict[model_name] = {
        'RMSE': rmse,
        'MAE': mae,
        'final_loss': train_losses[-1]
    }
    
    # Графік прогнозів
    plt.figure(figsize=(15, 6))
    
    # Графік 1: Прогнози vs Реальні значення
    plt.subplot(1, 2, 1)
    plt.plot(y_test_inv.flatten()[:100], label='Actual', alpha=0.8)
    plt.plot(preds_inv.flatten()[:100], label='Predicted', alpha=0.8)
    plt.title(f"{model_name} - Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Графік 2: Loss під час навчання
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.title(f"{model_name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print("-" * 50)

# Основний код для експериментів
results = {}

# Параметри для експериментів
window_sizes = [5, 10, 15]
hidden_sizes = [(64,), (128,), (64, 32), (128, 64), (256, 128)]
learning_rates = [0.001, 0.01]
epochs_list = [50, 100]

print("🚀 Початок експериментів з MLP моделями")
print("=" * 60)

# Основний цикл експериментів
experiment_count = 0
total_experiments = len(window_sizes) * len(hidden_sizes) * len(learning_rates) * len(epochs_list)

for window_size in window_sizes:
    print(f"\n📊 Window size: {window_size}")
    
    # Підготовка даних для поточного window_size
    X, y = prepare_data(scaled_df, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    input_size = X_train.shape[1]
    
    for hidden_config in hidden_sizes:
        for lr in learning_rates:
            for epochs in epochs_list:
                experiment_count += 1
                
                # Створення моделі
                if len(hidden_config) == 1:
                    model = MLPModel(input_size=input_size, 
                                   hidden_size1=hidden_config[0])
                    model_name = f"MLP_w{window_size}_h{hidden_config[0]}_lr{lr}_e{epochs}"
                else:
                    model = MLPModel(input_size=input_size, 
                                   hidden_size1=hidden_config[0],
                                   hidden_size2=hidden_config[1])
                    model_name = f"MLP_w{window_size}_h{hidden_config[0]}-{hidden_config[1]}_lr{lr}_e{epochs}"
                
                print(f"\n[{experiment_count}/{total_experiments}] {model_name}")
                
                # Навчання моделі
                run_torch_model(model_name, model, X_train, y_train, X_test, y_test, 
                              target_scaler, results, epochs=epochs, lr=lr)

# Результати
print("\n" + "=" * 80)
print("📈 ПІДСУМОК РЕЗУЛЬТАТІВ ВСІХ ЕКСПЕРИМЕНТІВ")
print("=" * 80)

# Сортування за RMSE
sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])

print(f"{'Model Name':<40} {'RMSE':<10} {'MAE':<10} {'Final Loss':<12}")
print("-" * 80)

for model_name, metrics in sorted_results:
    rmse = metrics['RMSE']
    mae = metrics['MAE']
    final_loss = metrics['final_loss']
    print(f"{model_name:<40} {rmse:<10.4f} {mae:<10.4f} {final_loss:<12.6f}")

# Найкраща модель
best_model = sorted_results[0]
print(f"\n🏆 НАЙКРАЩА МОДЕЛЬ: {best_model[0]}")
print(f"   RMSE: {best_model[1]['RMSE']:.4f}")
print(f"   MAE: {best_model[1]['MAE']:.4f}")

# Аналіз результатів по window_size
print(f"\n📊 АНАЛІЗ ПО WINDOW SIZE:")
for ws in window_sizes:
    ws_results = [(name, metrics) for name, metrics in results.items() if f"_w{ws}_" in name]
    if ws_results:
        best_ws = min(ws_results, key=lambda x: x[1]['RMSE'])
        avg_rmse = np.mean([metrics['RMSE'] for _, metrics in ws_results])
        print(f"Window {ws}: Найкращий RMSE = {best_ws[1]['RMSE']:.4f}, Середній RMSE = {avg_rmse:.4f}")