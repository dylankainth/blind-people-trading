import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

### Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, df_scaled, T, start_idx, end_idx, features, target='Close'):
        self.df_scaled = df_scaled
        self.T = T
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.features = features
        self.target = target

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    def __getitem__(self, idx):
        t = self.start_idx + idx
        seq = self.df_scaled.iloc[t - self.T:t][self.features].values
        target = self.df_scaled.iloc[t][self.target]
        return torch.FloatTensor(seq), torch.FloatTensor([target])

### Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        return attn_weights @ v

### Enhanced CNN-LSTM Model
class SolanaPredictor(nn.Module):
    def __init__(self, input_size, cnn_channels=64, lstm_hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            # Double CNN Layer
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(0.2)
        )
        self.se = nn.Sequential(
            nn.Linear(cnn_channels, cnn_channels // 2),
            nn.ReLU(),
            nn.Linear(cnn_channels // 2, cnn_channels),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = SelfAttention(lstm_hidden_size)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.permute(0, 2, 1)
        # Modified to add SE back in the game
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        se_input = cnn_out.mean(dim=1)
        se_weights = self.se(se_input).unsqueeze(1)
        scaled_features = cnn_out * se_weights
        lstm_out, _ = self.lstm(scaled_features)
        attn_out = self.attention(lstm_out)
        last_out = attn_out[:, -1, :]
        return self.fc(last_out)

### Data Preprocessing and Feature Engineering
def load_and_preprocess_data(ticker, period, interval, T, train_ratio, val_ratio):
    df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()
    # Debug Process to check the original df
    print("Printing Current df Structure...")
    print(df)

    # Log transformation
    df['Log_Close'] = np.log(df['Close'])  # Preserve log prices
    df['Close'] = df['Log_Close'].copy()   # 'Close' starts as log prices
    
    # Technical indicators
    # SMA_10
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    # EMA_10
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    # MACD Signal
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # TR
    df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
    # DX
    df['DX'] = 100 * (df['TR'].rolling(window=14).sum() / df['TR'].rolling(window=14).mean())
    # ADX
    df['ADX'] = df['DX'].rolling(window=14).mean()
    # AO
    df['AO'] = df['Close'].rolling(window=5).mean() - df['Close'].rolling(window=34).mean()
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    # VROC
    epsilon = 1e-10
    vroc = (df['Volume'] / (df['Volume'].shift(14) + epsilon) - 1) * 100
    df['VROC'] = vroc
    
    # Feature list
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI_14', 
                'MACD', 'MACD_signal', 'ADX', 'AO', 'OBV', 'VROC']
    df = df.dropna(subset=features).reset_index(drop=True)
    
    # Debug Process
    print("Printing df after adding technical indicators")
    print(df)

    # Stationarity check and differencing
    differenced = False
    result = adfuller(df['Close'])
    if result[1] > 0.05:
        df['Close'] = df['Close'].diff().dropna()
        df = df.dropna().reset_index(drop=True)
        differenced = True
    
    # Debug Process
    print("Printing df after adding adf test")
    print(df)
    
    # Scaling
    N = len(df)
    total_samples = N - T
    train_samples = int(train_ratio * total_samples)
    val_samples = int(val_ratio * total_samples)
    train_end = T + train_samples - 1

    feature_scaler = MinMaxScaler().fit(df.iloc[:train_end][features])
    target_scaler = MinMaxScaler().fit(df.iloc[T:train_end + 1][['Close']])

    df_scaled = df.copy()
    df_scaled[features] = feature_scaler.transform(df[features])
    df_scaled['Close'] = target_scaler.transform(df[['Close']])

    split_indices = {
        'train': (T, T + train_samples - 1),
        'val': (T + train_samples, T + train_samples + val_samples - 1),
        'test': (T + train_samples + val_samples, N - 1)
    }

    print("df after scaling")
    print(df_scaled)
    
    return df_scaled, feature_scaler, target_scaler, split_indices, features, differenced

### Helper Functions
def create_datasets_and_loaders(df_scaled, T, split_indices, features, batch_size):
    train_dataset = TimeSeriesDataset(df_scaled, T, *split_indices['train'], features)
    val_dataset = TimeSeriesDataset(df_scaled, T, *split_indices['val'], features)
    test_dataset = TimeSeriesDataset(df_scaled, T, *split_indices['test'], features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for seqs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                outputs = model(seqs)
                val_loss += criterion(outputs.squeeze(), targets.squeeze()).item()
        
        train_mse = train_loss / len(train_loader)
        val_mse = val_loss / len(val_loader)
        
        # Added MSE into Considering for Train and Val
        print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.8f} | Train MSE: {train_mse: 8f} | Val Loss: {val_loss:.8f} | Val MSE: {val_mse: .8f}')

def evaluate_model(model, test_loader, criterion, target_scaler):
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for seqs, targets in test_loader:
            outputs = model(seqs)
            test_preds.extend(outputs.squeeze().numpy())
            test_targets.extend(targets.squeeze().numpy())
    
    test_preds = target_scaler.inverse_transform(np.array(test_preds).reshape(-1, 1))
    test_targets = target_scaler.inverse_transform(np.array(test_targets).reshape(-1, 1))
    
    test_mse = mean_squared_error(test_targets, test_preds)
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(test_targets, test_preds)
    direction_pred = np.sign(np.diff(test_preds, axis=0))
    direction_true = np.sign(np.diff(test_targets, axis=0))
    directional_accuracy = np.mean(direction_pred == direction_true)
    
    return test_mse, test_rmse, test_mae, directional_accuracy

def benchmark_arima(df, split_indices):
    test_start, test_end = split_indices['test']
    train_data = df['Close'].iloc[:test_start]
    test_data = df['Close'].iloc[test_start:test_end + 1]
    model = ARIMA(train_data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))
    arima_mse = mean_squared_error(test_data, forecast)
    arima_rmse = math.sqrt(arima_mse)
    arima_mae = mean_absolute_error(test_data, forecast)
    return arima_mse, arima_rmse, arima_mae

def benchmark_naive_forecast(df_scaled, split_indices):
    """Benchmark using naive forecast (previous close as prediction)."""
    test_start, test_end = split_indices['test']
    naive_preds = df_scaled['Close'].iloc[test_start - 1:test_end].values
    naive_targets = df_scaled['Close'].iloc[test_start:test_end + 1].values
    naive_mse = mean_squared_error(naive_targets, naive_preds)
    naive_rmse = math.sqrt(naive_mse)
    naive_mae = mean_absolute_error(naive_targets, naive_preds)
    return naive_mse, naive_rmse, naive_mae

def benchmark_moving_average(df_scaled, split_indices, ma_window):
    """Benchmark using moving average."""
    test_start, test_end = split_indices['test']
    ma_preds = df_scaled['Close'].rolling(window=ma_window).mean().iloc[test_start:].values
    ma_targets = df_scaled['Close'].iloc[test_start:].values
    ma_mse = mean_squared_error(ma_targets[ma_window - 1:], ma_preds[ma_window - 1:])
    ma_rmse = math.sqrt(ma_mse)
    ma_mae = mean_absolute_error(ma_targets[ma_window - 1:], ma_preds[ma_window - 1:])
    return ma_mse, ma_rmse, ma_mae

def predict_next_hour(model, df_scaled, T, features, target_scaler, differenced):
    model.eval()
    with torch.no_grad():
        last_seq = torch.FloatTensor(df_scaled.iloc[-T:][features].values).unsqueeze(0)
        pred_scaled = model(last_seq).item()
        pred_diff_or_log = target_scaler.inverse_transform([[pred_scaled]])[0][0]
        if differenced:
            last_log_close = df_scaled['Log_Close'].iloc[-1]  # Last known log price (P_{t-1})
            pred_log_close = last_log_close + pred_diff_or_log  # P_t = P_{t-1} + ΔP_t
            pred_price = np.exp(pred_log_close)  # Close_t = exp(P_t)
        else:
            pred_price = np.exp(pred_diff_or_log)  # Close_t = exp(P_t)
    return pred_price

### Main Execution
if __name__ == "__main__":
    ticker = 'SOL-USD'
    period = "730d"
    interval = '1h'
    T = 24
    train_ratio = 0.7
    val_ratio = 0.15
    num_epochs = 15
    ma_window = 24
    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]
    best_params = {'lr': 0.0005, 'batch_size': 32}

    # Load data
    df_scaled, feature_scaler, target_scaler, split_indices, features, differenced = load_and_preprocess_data(
        ticker, period, interval, T, train_ratio, val_ratio
    )

    # Dataset and loaders (unchanged)
    train_loader, val_loader, test_loader = create_datasets_and_loaders(
        df_scaled, T, split_indices, features, best_params['batch_size']
    )
    model = SolanaPredictor(input_size=len(features))
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Evaluation (unchanged)
    test_mse, test_rmse, test_mae, directional_accuracy = evaluate_model(model, test_loader, criterion, target_scaler)
    print(f'Test MSE: {test_mse:.8f} | Test RMSE: {test_rmse:.8f} | Test MAE: {test_mae:.8f} | Directional Accuracy: {directional_accuracy:.8f}')

    # Benchmarks (unchanged)
    arima_mse, arima_rmse, arima_mae = benchmark_arima(df_scaled, split_indices)
    print(f'ARIMA - MSE: {arima_mse:.8f} | RMSE: {arima_rmse:.8f} | MAE: {arima_mae:.8f}')
    naive_mse, naive_rmse, naive_mae = benchmark_naive_forecast(df_scaled, split_indices)
    print(f'Naive Forecast - MSE: {naive_mse:.8f} | RMSE: {naive_rmse:.8f} | MAE: {naive_mae:.8f}')
    ma_mse, ma_rmse, ma_mae = benchmark_moving_average(df_scaled, split_indices, ma_window)
    print(f'Moving Average ({ma_window}h) - MSE: {ma_mse:.8f} | RMSE: {ma_rmse:.8f} | MAE: {ma_mae:.8f}')

    # Prediction
    pred_price = predict_next_hour(model, df_scaled, T, features, target_scaler, differenced)
    print(f'Predicted next hour Close price: {pred_price:.2f} USD')