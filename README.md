# Solana Price Direction Predictor

## Introduction

This script predicts the direction of the next hour's price movement for Solana (SOL-USD) using a machine learning model that combines CNNs, LSTMs, and attention mechanisms. The model is trained to predict both the value and direction of the price change, with a focus on directional accuracy.

## Algorithm Overview

1. **Data Preparation**:
   - Download historical data using yfinance.
   - Clean extreme values and apply log transformation.
   - Calculate technical indicators (SMA, EMA, RSI, etc.).
   - Ensure stationarity with differencing if necessary.
   - Scale features and target using RobustScaler and MinMaxScaler.

2. **Model Training**:
   - Initialize multiple `SolanaPredictor` models with different seeds.
   - Train each model using the `DirectionalLoss` function.
   - Employ learning rate scheduling and early stopping based on validation directional accuracy.

3. **Ensemble Prediction**:
   - Average predictions from the ensemble of models.
   - Determine the direction (UP or DOWN) by comparing the predicted price to the current price.

## Data Preprocessing

- **Cleaning**: Clip extreme values to prevent training issues.
- **Transformation**: Log-transform closing prices.
- **Indicators**: Compute SMA, EMA, RSI, MACD, ADX, AO, OBV, VROC.
- **Stationarity**: Use Augmented Dickey-Fuller test and differencing.
- **Scaling**: Apply RobustScaler to features and MinMaxScaler to the target.
- **Directional Features**: Add momentum, volatility, and breakout signals.

## Model Architecture

The `SolanaPredictor` model includes:

- **CNN**: Extracts features from the input sequence.
- **LSTM Heads**: Separate LSTMs for value and direction predictions.
- **Self-Attention**: Weighs time steps using:
```math
\text{attn\_weights} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)
```
- **Time-based Attention**: Uses positional encoding to emphasize recent data.

## Loss Function

The `DirectionalLoss` combines:

- **Value Loss**: Huber loss for robustness.
- **Direction Loss**: Penalizes incorrect direction predictions with class weights.

Total loss:
```math
\text{loss} = w_v \cdot \text{HuberLoss}(\text{pred}, \text{target}) + w_d \cdot \text{direction\_loss}
```

## Ensemble Method

An ensemble of three models is used to enhance prediction stability and accuracy.

## Training Process

- **Optimizer**: Adam with weight decay.
- **Scheduler**: Reduces learning rate on plateau.
- **Early Stopping**: Monitors validation directional accuracy.

## Prediction

The ensemble model predicts the next hour's price, and the direction is determined by comparing it to the current price.

## Mathematical Foundations with Explanations

### 1. Data Preprocessing

**1.1 Log Transformation**  
*Purpose: Stabilize variance in price series*  
```math
\text{Log\_Close} = \ln(\text{Close})
```
Applies natural logarithm to closing prices to:
- Reduce sensitivity to extreme values
- Convert multiplicative relationships to additive
- Prepare for possible differencing

**1.2 Differencing**  
*Purpose: Achieve stationarity*  
```math
\text{Close}_t = \text{Log\_Close}_t - \text{Log\_Close}_{t-1}
```
If Augmented Dickey-Fuller test shows non-stationarity (p-value > 0.05), this transformation:
- Removes trend components
- Makes statistical properties constant over time
- Required for ARIMA-style modeling

**1.3 Technical Indicators (RSI Example)**  
*Purpose: Capture momentum*  
```math
\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Avg Gain}_{14}}{\text{Avg Loss}_{14}}}
```
- Computes relative strength over 14-period window
- Values > 70 indicate overbought, < 30 oversold
- Avg Gain/Loss are exponential moving averages

**1.4 Price Momentum**  
*Purpose: Identify trend direction*  
```math
\text{Price\_Momentum}_n = \text{Close}_t - \text{Close}_{t-n}
```
- Calculates n-period price change (typically n=4 for 4-hour momentum)
- Positive values indicate upward trend
- Helps model recognize acceleration/deceleration

---

### 2. Model Architecture

**2.1 LSTM Hidden State Update**  
*Purpose: Maintain temporal memory*  
```math
h_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})
```
- $h_t \in \mathbb{R}^d$: Hidden state vector at time $$t$$
- $c_t \in \mathbb{R}^d$: Cell state vector carrying long-term information  
- $d$: Hidden dimension size
- Update gate controls information retention:

**2.2 Self-Attention Mechanism**  
*Purpose: Focus on relevant time steps*
```math
\left( \sum_{k=1}^{n} a_k b_k \right)^2 \leq \left( \sum_{k=1}^{n} a_k^{2} \right) \left( \sum_{k=1}^{n} b_k^{2} \right)
```

**2.3 Time-Based Attention**  
*Purpose: Prioritize recent patterns*  
```math
\left( \sum_{k=1}^{n} a_k b_k \right)^2 \leq \left( \sum_{k=1}^{n} a_k^{2} \right)\left( \sum_{k=1}^{n} b_k^{2} \right)
```
- Learnable time weights ($$time\_weights$$) adaptively emphasize recent data
- Combined with positional encoding for temporal context

**2.4 Value Adjustment**  
*Purpose: Couple price/direction predictions*  
```math
\text{Value\_Adjusted} = \text{Value\_Pred} \cdot \left(0.8 + 0.2 \cdot \tanh(\text{Dir\_Logit})\right)
```
- $\tanh$ bounds adjustment between [-0.2, +0.2]
- 0.8 base weight maintains stability
- Direction logit ($Dir\_Logit$) from classification head

---

### 3. Loss Functions

**3.1 Directional Loss**  
*Purpose: Joint optimization*  
```math
L = 0.6 \cdot \text{HuberLoss} + 0.4 \cdot \text{DirectionLoss}
```
- 60% weight on value accuracy ($w_v=0.6$)
- 40% weight on direction correctness ($w_d=0.4$)
- Balances regression and classification objectives

**3.2 Huber Loss**  
*Purpose: Robust regression*  
```math
\text{HuberLoss} = 
\begin{cases} 
  \frac{1}{2}(\hat{y} - y)^2 & \text{if } |\hat{y} - y| \leq \delta \\
  \delta|\hat{y} - y| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
```
  
- $\delta=1.0$: Threshold for linear/quadratic behavior
- Less sensitive to outliers than MSE
- Smooth transition at threshold

**3.3 Direction Loss**  
*Purpose: Handle class imbalance*  
```math
\text{DirectionLoss} = \frac{\sum w_i \cdot \mathbf{1}(\text{sign}(\hat{y}_i) \neq \text{sign}(y_i))}{\sum \mathbf{1}(|y_i| > \epsilon)}
```
- $w_i$: Instance weight (higher for large price moves)
- $\epsilon=0.005$: Filter for significant movements
- Focuses model on meaningful directional changes

---

### 4. Ensemble Prediction

**4.1 Model Combination**  
*Purpose: Improve stability*  
```math
\hat{y}_{\text{ensemble}} = \frac{1}{3}\sum_{i=1}^3 \hat{y}_i
```
- Simple average of 3 independently trained models
- Reduces variance through diversification
- Each model uses different random initialization

**4.2 Price Reconstruction**  
*Purpose: Convert predictions to USD*  
```math
\text{Pred\_Price} = \exp(\text{Last\_Log\_Close} + \text{Pred\_Diff})
```
1. If differenced: Reconstruct log price by adding prediction to last observed value
2. Exponentiate to reverse log transformation
3. Convert stationary value back to USD price

---

### 5. Training Dynamics

**5.1 Learning Rate Scheduling**  
*Adaptation Rule*:
```math
\text{lr}_{\text{new}} = \text{lr}_{\text{old}} \cdot 0.5 \quad \text{if val\_accuracy plateau > 5 epochs}
```
- Initial learning rate: 0.0005
- Factor: 0.5 reduction on plateaus
- Prevents overshooting in loss landscape

**5.2 Directional Prediction**  
*Decision Rule*:
```math
\text{Direction} = 
\begin{cases}
  \text{"UP"} & \text{if } \hat{y}_{\text{ensemble}} > \text{Current\_Price} \\
  \text{"DOWN"} & \text{otherwise}
\end{cases}
```
  
- Compares prediction to most recent observed price
- Threshold-free decision for operational simplicity
- Final output combines price + direction

---

## Implementation Flow in Raw
```mermaid
A[Raw Price Data] --> B[Log Transform]
B --> C{Stationary?}
C -->|Yes| D[Feature Engineering]
C -->|No| E[Differencing]
E --> D
D --> F[Technical Indicators]
F --> G[CNN Feature Extraction]
G --> H[LSTM Temporal Modeling]
H --> I[Self-Attention Context]
I --> J[Time-Weighted Attention]
J --> K[Value Head]
J --> L[Direction Head]
K & L --> M[Ensemble Averaging]
M --> N[Price Reconstruction]
N --> O[Direction Decision]
```

## Research Paper

Included in Solana_Price_Prediction_Model Research_Paper.pdf


# Multi-Agent System for Solana Price Prediction

## Abstract

This paper introduces the multi-agent architecture implemented in the Solana Price Direction Predictor system. The framework leverages AutoGen's collaborative agent design to enhance prediction capabilities through specialized agents with distinct roles. We detail the agent communication patterns, function calling mechanisms, and integration with the CNN-LSTM prediction model.

## 1. Introduction

Modern financial prediction systems benefit from collaborative intelligence where specialized components work in concert. Our system implements a multi-agent design pattern where autonomous agents with distinct capabilities collaborate to improve prediction accuracy for Solana price movements.

## 2. Multi-Agent Architecture

### 2.1 Agent Composition

The system employs a group chat structure with the following agents:

- **UserProxy**: Represents the end-user, initiates requests and presents final results
- **FinancialAnalyst**: Interprets market conditions and technical indicators
- **DataScientist**: Processes raw data and executes model predictions
- **MarketSentimentAnalyst**: Analyzes news and social media signals
- **Coordinator**: Orchestrates workflow and synthesizes insights from other agents

### 2.2 Communication Flow

```mermaid
graph TD
    A[UserProxy] -->|Request Prediction| B[Coordinator]
    B -->|Request Data| C[DataScientist]
    B -->|Request Analysis| D[FinancialAnalyst]
    B -->|Request Sentiment| E[MarketSentimentAnalyst]
    C -->|Data & Predictions| B
    D -->|Technical Analysis| B
    E -->|Sentiment Score| B
    B -->|Synthesized Prediction| A
```

### 2.3 Asynchronous Processing

The system runs agent collaboration asynchronously using Python's asyncio:

```python
# Initialization in FastAPI route
asyncio.create_task(run_chat_async())
```

This allows the API to remain responsive while the intensive multi-agent discussion occurs in the background.

## 3. Function Calling Mechanism

### 3.1 Function Registration

Each agent exposes callable functions using a standardized schema:

```python
functions = [
    {
        "name": "fetch_price_data",
        "description": "Fetch historical price data for a given symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., SOL-USD)"},
                "interval": {"type": "string", "description": "Data interval (e.g., 1h)"},
                "range": {"type": "string", "description": "Predefined date range"}
            },
            "required": ["symbol"]
        }
    },
    # Additional functions...
]
```

### 3.2 Cross-Agent Function Invocation

Agents can request services from each other using function calls:

```python
async def run_prediction(symbol, interval="1h", range="5d"):
    # Collect data
    price_data = await fetch_price_data(symbol, interval, range)
    
    # Generate features
    features = await calculate_technical_indicators(price_data)
    
    # Make prediction
    prediction = await predict_price_direction(features)
    
    return prediction
```

### 3.3 Integration with Yahoo Finance API

The system leverages FastAPI endpoints to retrieve real-time and historical data:

```python
async def fetch_price_data(symbol, interval, range):
    api_url = f"/price?symbol={symbol}&interval={interval}&range={range}"
    response = await httpx.AsyncClient().get(api_url)
    return response.json()
```

## 4. Agent Specialization

### 4.1 DataScientist Agent

- Retrieves historical price data via API calls
- Executes preprocessing pipeline (log transformation, differencing, scaling)
- Runs the CNN-LSTM-Attention model prediction
- Returns both quantitative predictions and confidence scores

### 4.2 FinancialAnalyst Agent

- Interprets technical indicators (RSI, MACD, etc.)
- Identifies support/resistance levels
- Evaluates trading volume patterns
- Provides price target ranges based on technical analysis

### 4.3 MarketSentimentAnalyst Agent

- Analyzes news articles and social media sentiment
- Quantifies market sentiment as numerical scores
- Identifies emerging narratives affecting Solana
- Evaluates correlation between sentiment and price action

## 5. Decision Synthesis

The Coordinator agent employs a weighted decision framework:

```python
def synthesize_prediction(model_pred, technical_analysis, sentiment_score):
    # Base weights
    w_model = 0.6
    w_technical = 0.3
    w_sentiment = 0.1
    
    # Adjust based on model confidence
    if model_pred['confidence'] < 0.6:
        w_model = 0.4
        w_technical = 0.4
        w_sentiment = 0.2
    
    # Calculate direction probability
    up_probability = (
        w_model * model_pred['up_prob'] +
        w_technical * technical_analysis['bullish_score'] +
        w_sentiment * sentiment_score['positive_ratio']
    )
    
    return {
        "direction": "UP" if up_probability > 0.5 else "DOWN",
        "confidence": abs(up_probability - 0.5) * 2,
        "price_target": model_pred['price']
    }
```

## 6. Conclusion

The multi-agent architecture provides several advantages over monolithic prediction systems:

1. **Specialization**: Each agent focuses on specific aspects of the prediction problem
2. **Robustness**: Failure in one component doesn't disable the entire system
3. **Explainability**: Decision process is transparent through agent conversations
4. **Adaptability**: New agents can be integrated without redesigning the system

This collaborative intelligence approach enhances the Solana price prediction system by combining machine learning forecasts with technical analysis and market sentiment.