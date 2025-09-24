#!/usr/bin/env python3
# Time Series Learning Project

Basic forecasting suite with classical models and evaluation.

## Models
- Naive/Seasonal Naive Baselines (`baselines.py`)
- ARIMA (`arima.py`)
- Exponential Smoothing (Holt-Winters) (`holtwinters.py`)

## Usage
```
python main.py --dataset dataset/air_passengers.csv --date_col Month --value_col Passengers --freq M
```

If dataset missing, an example seasonal series is generated.

## Metrics & Plots
- Metrics: MAE, RMSE, MAPE on test split
- Plots: Train/test split, forecasts vs actuals, residuals


