This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

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
  \[
  \text{attn\_weights} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)
  \]
- **Time-based Attention**: Uses positional encoding to emphasize recent data.

## Loss Function

The `DirectionalLoss` combines:

- **Value Loss**: Huber loss for robustness.
- **Direction Loss**: Penalizes incorrect direction predictions with class weights.

Total loss:
\[
\text{loss} = w_v \cdot \text{HuberLoss}(pred, target) + w_d \cdot \text{direction\_loss}
\]

## Ensemble Method

An ensemble of three models is used to enhance prediction stability and accuracy.

## Training Process

- **Optimizer**: Adam with weight decay.
- **Scheduler**: Reduces learning rate on plateau.
- **Early Stopping**: Monitors validation directional accuracy.

## Prediction

The ensemble model predicts the next hour's price, and the direction is determined by comparing it to the current price.

