import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller

def run():
    # loss function
    class DirectionalLoss(nn.Module):
        def __init__(self, value_weight=0.7, direction_weight=0.3, class_weights=None):
            super().__init__()
            self.value_loss = nn.HuberLoss()
            self.value_weight = value_weight
            self.direction_weight = direction_weight
            self.class_weights = class_weights
            
        def forward(self, pred, target):
            # Value component
            value_loss = self.value_loss(pred, target)
            
            # Direction component with class weights
            pred_direction = torch.sign(pred)
            target_direction = torch.sign(target)
            valid_mask = (torch.abs(target) > 1e-6)
            
            if valid_mask.sum() > 0:
                # Apply class weights to direction loss
                if self.class_weights is not None:
                    weights = torch.ones_like(target)
                    for cls, weight in self.class_weights.items():
                        weights[target_direction == cls] = weight
                    direction_loss = (((pred_direction != target_direction) & valid_mask).float() * weights).mean()
                else:
                    direction_loss = ((pred_direction != target_direction) & valid_mask).float().mean()
            else:
                direction_loss = torch.tensor(0.0, device=pred.device)
            
            return self.value_weight * value_loss + self.direction_weight * direction_loss

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

    ### Enhanced CNN-LSTM Model - Updated to match CryptoDirectionPredictor architecture
    class SolanaPredictor(nn.Module):
        def __init__(self, input_size, cnn_channels=64, lstm_hidden_size=128):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(cnn_channels),
                nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(cnn_channels),
                nn.Dropout(0.2)
            )
            
            # Separate heads for value and direction
            self.lstm_value = nn.LSTM(cnn_channels, lstm_hidden_size, batch_first=True)
            self.lstm_dir = nn.LSTM(cnn_channels, lstm_hidden_size, batch_first=True)
            
            # Attention mechanism with recent data focus
            self.time_weights = nn.Parameter(torch.ones(24) / 24)  # Initialize with uniform weights
            
            # Recent data gets higher weight through positional encoding
            position = torch.arange(24).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, 8, 2).float() * (-math.log(10000.0) / 8))
            pos_enc = torch.zeros(24, 8)
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_enc', pos_enc)
            
            # Output heads
            self.fc_value = nn.Linear(lstm_hidden_size, 1)
            self.fc_dir = nn.Linear(lstm_hidden_size, 1)
            
        def forward(self, x):
            batch_size, seq_len, feat_dim = x.size()
            
            # Apply time-based attention (focus on recent data)
            time_weights = F.softmax(self.time_weights, dim=0)
            weighted_input = x * time_weights.view(1, -1, 1)
            
            # CNN feature extraction
            cnn_input = weighted_input.permute(0, 2, 1)
            cnn_out = self.cnn(cnn_input)
            cnn_out = cnn_out.permute(0, 2, 1)
            
            # Add positional encoding - fix size mismatch by using only the first feat_dim columns
            use_dims = min(self.pos_enc.size(1), cnn_out.size(2))
            pos_enc_trimmed = self.pos_enc[:, :use_dims]
            
            # Only add positional encoding to the dimensions that match
            if cnn_out.size(2) >= use_dims:
                cnn_out[:, :, :use_dims] = cnn_out[:, :, :use_dims] + pos_enc_trimmed.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Value head
            value_out, _ = self.lstm_value(cnn_out)
            value_pred = self.fc_value(value_out[:, -1, :])
            
            # Direction head
            dir_out, _ = self.lstm_dir(cnn_out)
            dir_logit = self.fc_dir(dir_out[:, -1, :])
            
            # Combine predictions (use direction to adjust value)
            dir_sign = torch.tanh(dir_logit)  # Soft sign between -1 and 1
            value_adjusted = value_pred * (0.8 + 0.2 * dir_sign)  # Adjust magnitude by direction
            
            return value_adjusted

    class EnsemblePredictor:
        def __init__(self, models, weights=None):
            self.models = models
            self.weights = weights if weights is not None else [1/len(models)] * len(models)
            
        def predict(self, x):
            predictions = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(x)
                    predictions.append(pred)
                    
            # Weighted average of predictions
            ensemble_pred = torch.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += pred * self.weights[i]
                
            return ensemble_pred

    def analyze_target_distribution(df, target_col='Close'):
        """Analyze and visualize the target distribution to identify class imbalance."""
        directions = np.sign(df[target_col].diff())
        up_count = np.sum(directions > 0)
        down_count = np.sum(directions < 0)
        flat_count = np.sum(directions == 0)
        
        # Calculate class weights for balanced sampling
        weights = {
            1: 1.0 / (up_count / len(directions)) if up_count > 0 else 0,
            -1: 1.0 / (down_count / len(directions)) if down_count > 0 else 0,
            0: 1.0 / (flat_count / len(directions)) if flat_count > 0 else 0
        }
        
        return weights

    def clean_extreme_values(df, columns, threshold=10):
        """Replace extreme values with reasonable limits to prevent training issues."""
        for col in columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    
                # Clip values
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df

    ### Data Preprocessing and Feature Engineering
    def add_directional_features(df):
        """Add features specifically designed for direction prediction."""
        # Price momentum features
        df['Price_Momentum_5'] = df['Close'].diff(5)
        df['Price_Momentum_10'] = df['Close'].diff(10)
        
        # Directional change features
        df['Dir_Change_5'] = np.sign(df['Close'].diff(5))
        df['Dir_Change_10'] = np.sign(df['Close'].diff(10))
        
        # Volatility features
        df['Volatility_5'] = df['Close'].rolling(5).std()
        df['Volatility_10'] = df['Close'].rolling(10).std()
        
        # Price reversal signals
        if 'RSI_14' in df.columns:
            df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(float)
            df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(float)
        
        # Volume surge detection
        if 'Volume' in df.columns:
            df['Volume_Surge'] = (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5).astype(float)
        
        # Breakout detection
        df['Upper_Band'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
        df['Lower_Band'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
        df['Breakout_Up'] = (df['Close'] > df['Upper_Band'].shift(1)).astype(float)
        df['Breakout_Down'] = (df['Close'] < df['Lower_Band'].shift(1)).astype(float)
        
        # Additional features for crypto specifically
        if 'Volume' in df.columns:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df['Close_VWAP_Ratio'] = df['Close'] / df['VWAP']
        
        return df

    def load_and_preprocess_data(ticker, period, interval, T, train_ratio, val_ratio):
        df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()
        # Clean extreme values before any transformations
        df = clean_extreme_values(df, ['Volume', 'Open', 'High', 'Low', 'Close'], threshold=5)

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
        
        # Clean any extreme values that might have been created
        df = clean_extreme_values(df, ['OBV', 'VROC', 'DX', 'ADX'], threshold=10)
        
        # Feature list
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI_14', 
                    'MACD', 'MACD_signal', 'ADX', 'AO', 'OBV', 'VROC']
        df = df.dropna(subset=features).reset_index(drop=True)
        
        # Add directional features
        df = add_directional_features(df)
        # Stationarity check and differencing
        differenced = False
        result = adfuller(df['Close'])
        if result[1] > 0.05:
            df['Close'] = df['Close'].diff().dropna()
            df = df.dropna().reset_index(drop=True)
            differenced = True
        
        # Scaling
        N = len(df)
        total_samples = N - T
        train_samples = int(train_ratio * total_samples)
        val_samples = int(val_ratio * total_samples)
        train_end = T + train_samples - 1

        all_features = [col for col in df.columns if col not in ['Date', 'Datetime', 'index']]
        
        feature_scaler = RobustScaler().fit(df.iloc[:train_end][all_features])
        target_scaler = MinMaxScaler().fit(df.iloc[T:train_end + 1][['Close']])

        df_scaled = df.copy()
        df_scaled[all_features] = feature_scaler.transform(df[all_features])
        df_scaled['Close'] = target_scaler.transform(df[['Close']])
        df_scaled['Original_Log_Close'] = df['Log_Close'].copy()  # Add original log prices

        split_indices = {
            'train': (T, T + train_samples - 1),
            'val': (T + train_samples, T + train_samples + val_samples - 1),
            'test': (T + train_samples + val_samples, N - 1)
        }
        return df_scaled, feature_scaler, target_scaler, split_indices, all_features, differenced

    ### Helper Functions
    def create_datasets_and_loaders(df_scaled, T, split_indices, features, batch_size):
        train_dataset = TimeSeriesDataset(df_scaled, T, *split_indices['train'], features)
        val_dataset = TimeSeriesDataset(df_scaled, T, *split_indices['val'], features)
        test_dataset = TimeSeriesDataset(df_scaled, T, *split_indices['test'], features)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, target_scaler, scheduler=None, patience=5):
        best_dir_acc = 0.0
        patience_counter = 0
        
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
            val_preds, val_targets = [], []
            with torch.no_grad():
                for seqs, targets in val_loader:
                    outputs = model(seqs)
                    val_loss += criterion(outputs.squeeze(), targets.squeeze()).item()
                    val_preds.extend(outputs.squeeze().numpy())
                    val_targets.extend(targets.squeeze().numpy())
            
            # Calculate directional accuracy on validation set
            val_preds = np.array(val_preds).reshape(-1, 1)
            val_targets = np.array(val_targets).reshape(-1, 1)
            val_preds_original = target_scaler.inverse_transform(val_preds)
            val_targets_original = target_scaler.inverse_transform(val_targets)
            
            val_dir_pred = np.sign(val_preds_original[:, 0])
            val_dir_true = np.sign(val_targets_original[:, 0])
            valid_indices = np.abs(val_targets_original[:, 0]) > 1e-6
            val_dir_acc = np.mean(val_dir_pred[valid_indices] == val_dir_true[valid_indices]) if np.sum(valid_indices) > 0 else 0.0
            
            train_mse = train_loss / len(train_loader)
            val_mse = val_loss / len(val_loader)
            
            # Update learning rate scheduler if provided
            if scheduler is not None:
                scheduler.step(val_dir_acc)
            
            # Early stopping based on directional accuracy
            if val_dir_acc > best_dir_acc:
                best_dir_acc = val_dir_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))

    def predict_next_hour_ensemble(ensemble, df_scaled, T, features, target_scaler, differenced):
        with torch.no_grad():
            last_seq = torch.FloatTensor(df_scaled.iloc[-T:][features].values).unsqueeze(0)
            pred_scaled = ensemble.predict(last_seq).item()
            pred_diff_or_log = target_scaler.inverse_transform([[pred_scaled]])[0][0]
            if differenced:
                last_log_close = df_scaled['Original_Log_Close'].iloc[-1]  # Use original log price
                pred_log_close = last_log_close + pred_diff_or_log
                pred_price = np.exp(pred_log_close)
            else:
                pred_price = np.exp(pred_diff_or_log)
        return pred_price

    ### Main Execution
        # Configuration
    ticker = 'SOL-USD'
    period = "730d"
    interval = '1h'
    T = 24  # Look-back window
    train_ratio = 0.7
    val_ratio = 0.15
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.0005
        
        # 6. Fix Data Leakage Concerns & 7. Anomaly Fixes for Extreme Values
        # These are addressed in load_and_preprocess_data() which includes clean_extreme_values()
    df_scaled, feature_scaler, target_scaler, split_indices, features, differenced = load_and_preprocess_data(
            ticker, period, interval, T, train_ratio, val_ratio
        )
        
        # 1. Address Class Imbalance - Analyze target distribution to calculate class weights
    class_weights = analyze_target_distribution(df_scaled, 'Close')
        
        # Create dataset and loaders
    train_loader, val_loader, test_loader = create_datasets_and_loaders(
            df_scaled, T, split_indices, features, batch_size
        )
        
        # 5. Cross-Timeframe Analysis
        # Additional timeframes for ensemble model
    timeframes = [
            {'period': "730d", 'interval': '1h'},
            {'period': "365d", 'interval': '1h'},
            {'period': "180d", 'interval': '1h'}
        ]
        
        # 4. Ensemble Approach for Better Stability
    models = []
    for i in range(3):  # Create 3 models with different seeds
            torch.manual_seed(42 + i)  # Different seed for initialization
            
            # 3. Model Architecture with Attention to Recent Data
            # SolanaPredictor already includes self-attention mechanisms
            model = SolanaPredictor(input_size=len(features))
            
            # Define loss with directional component
            criterion = DirectionalLoss(value_weight=0.6, direction_weight=0.4, class_weights=class_weights)
            
            # 8. Learning Rate Schedule
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
            
            # Train with scheduler for learning rate adjustment
            train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, 
                    target_scaler, scheduler=scheduler, patience=5)
            
            models.append(model)
        
        # Create ensemble predictor with equal weights
    ensemble = EnsemblePredictor(models)

    ensemble_pred = predict_next_hour_ensemble(ensemble, df_scaled, T, features, target_scaler, differenced)
    current_price = np.exp(df_scaled['Original_Log_Close'].iloc[-1])  # Use original log price
    ensemble_direction = "UP" if ensemble_pred > current_price else "DOWN"
    return ensemble_pred, ensemble_direction