import joblib
import torch
import yfinance as yf
import numpy as np
from statsmodels.tsa.stattools import adfuller
from model_final import SolanaPredictor, EnsemblePredictor, clean_extreme_values


def add_directional_features(df):
    """EXACT replica of training version"""
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
        df['Volume_Surge'] = (df['Volume'] > df['Volume'].rolling(
            10).mean() * 1.5).astype(float)

    # Breakout detection
    df['Upper_Band'] = df['Close'].rolling(
        20).mean() + (df['Close'].rolling(20).std() * 2)
    df['Lower_Band'] = df['Close'].rolling(
        20).mean() - (df['Close'].rolling(20).std() * 2)
    df['Breakout_Up'] = (df['Close'] > df['Upper_Band'].shift(1)).astype(float)
    df['Breakout_Down'] = (
        df['Close'] < df['Lower_Band'].shift(1)).astype(float)

    # Crypto-specific features
    if 'Volume' in df.columns:
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / \
            df['Volume'].cumsum()
        df['Close_VWAP_Ratio'] = df['Close'] / df['VWAP']

    return df


def preprocess_new_data(raw_df, differenced, all_features, T):
    """EXACT replica of training preprocessing"""
    df = raw_df.copy()

    # 1. Initial cleaning
    df = clean_extreme_values(
        df, ['Volume', 'Open', 'High', 'Low', 'Close'], threshold=5)

    # 2. Log transform
    df['Log_Close'] = np.log(df['Close'])
    df['Close'] = df['Log_Close'].copy()

    # 3. Technical indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain/loss))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['TR'] = df[['High', 'Low', 'Close']].max(
        axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
    df['DX'] = 100 * (df['TR'].rolling(14).sum() / df['TR'].rolling(14).mean())
    df['ADX'] = df['DX'].rolling(14).mean()
    df['AO'] = df['Close'].rolling(5).mean() - df['Close'].rolling(34).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['VROC'] = (df['Volume']/(df['Volume'].shift(14)+1e-10) - 1) * 100

    # 4. Clean indicator extremes
    df = clean_extreme_values(df, ['OBV', 'VROC', 'DX', 'ADX'], threshold=10)

    # 5. Add directional features
    df = add_directional_features(df)

    # 6. Stationarity handling
    if differenced:
        result = adfuller(df['Close'])
        if result[1] > 0.05:
            df['Close'] = df['Close'].diff().dropna()
            df = df.dropna().reset_index(drop=True)
            differenced = True

    # 7. Final cleanup
    df = df.dropna().reset_index(drop=True)

    # Ensure feature order matches training
    return df[all_features]


def run():
    # Load artifacts
    config = joblib.load(
        '../quant_prediction/config.pkl')
    feature_scaler = joblib.load('../quant_prediction/feature_scaler.pkl')
    target_scaler = joblib.load('../quant_prediction/target_scaler.pkl')

    # Load models with EXACT feature dimensions
    models = []
    for i in range(3):
        model = SolanaPredictor(input_size=len(config['all_features']))
        model.load_state_dict(torch.load(
            f'../quant_prediction/best_model_{i}.pth'))
        model.eval()
        models.append(model)
    ensemble = EnsemblePredictor(models)

    # Preprocess new data with training-compatible pipeline
    raw_df = yf.Ticker('SOL-USD').history(period='730d',
                                          interval='1h').reset_index()
    processed_df = preprocess_new_data(
        raw_df,
        config['differenced'],
        config['all_features'],
        config['T']
    )

    # Apply scaling
    df_scaled = processed_df.copy()
    df_scaled[config['all_features']] = feature_scaler.transform(
        processed_df[config['all_features']])
    df_scaled['Original_Log_Close'] = processed_df['Log_Close'].copy()

    # Generate prediction
    last_seq = torch.FloatTensor(
        df_scaled.iloc[-config['T']:][config['all_features']].values).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = ensemble.predict(last_seq).item()

    # Inverse transform
    pred_diff_or_log = target_scaler.inverse_transform([[pred_scaled]])[0][0]

    # Calculate final price
    if config['differenced']:
        last_log_price = df_scaled['Original_Log_Close'].iloc[-1]
        pred_price = np.exp(last_log_price + pred_diff_or_log)
    else:
        pred_price = np.exp(pred_diff_or_log)

    current_price = np.exp(df_scaled['Original_Log_Close'].iloc[-1])
    pre_direction = "UP" if pred_price > current_price else "DOWN"

    return {
        'pred_price': pred_price,
        'current_price': current_price,
        'pre_direction': pre_direction
    }
