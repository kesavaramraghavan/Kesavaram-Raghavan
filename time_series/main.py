#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.baselines import NaiveBaseline, SeasonalNaiveBaseline
from models.arima import ARIMAModel
from models.holtwinters import HoltWintersModel


def _generate_example_series(n=120, season_length=12, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 10 + 5 * np.sin(2 * np.pi * t / season_length)
    trend = 0.1 * t
    noise = rng.normal(0, 1.0, size=n)
    return trend + seasonal + noise


def train_test_split_series(y: np.ndarray, test_size=24):
    return y[:-test_size], y[-test_size:]


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


class TSAnalysis:
    def __init__(self, dataset_path: str, date_col: str, value_col: str, freq: Optional[str] = None):
        self.dataset_path = dataset_path
        self.date_col = date_col
        self.value_col = value_col
        self.freq = freq
        self.models = {}
        self.results = {}

    def load_series(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            y = _generate_example_series()
            dates = pd.date_range(start='2000-01-01', periods=len(y), freq=self.freq or 'M')
            df = pd.DataFrame({self.date_col: dates, self.value_col: y})
            df.to_csv(self.dataset_path, index=False)
        df = pd.read_csv(self.dataset_path, parse_dates=[self.date_col])
        if self.freq:
            df = df.set_index(self.date_col).asfreq(self.freq).reset_index()
        y = df[self.value_col].astype(float).values
        self.df = df
        self.y_train, self.y_test = train_test_split_series(y)
        return self

    def initialize_models(self):
        self.models = {
            'Naive': NaiveBaseline(),
            'SeasonalNaive': SeasonalNaiveBaseline(season_length=12),
            'ARIMA': ARIMAModel(order=(1, 1, 1)),
            'HoltWinters': HoltWintersModel(seasonal_periods=12),
        }
        return self

    def fit_and_forecast(self):
        horizon = len(self.y_test)
        for name, model in self.models.items():
            model.fit(self.y_train)
            y_fore = model.forecast(horizon)
            self.results[name] = {
                'y_fore': np.asarray(y_fore, dtype=float),
                'mae': mae(self.y_test, y_fore),
                'rmse': rmse(self.y_test, y_fore),
                'mape': mape(self.y_test, y_fore),
            }
        return self

    def select_best_models(self, n_best=3):
        sorted_items = sorted(self.results.items(), key=lambda x: x[1]['rmse'])
        self.best_models = sorted_items[:n_best]
        return self

    def plot_forecasts(self):
        plt.figure(figsize=(12, 6))
        n_train = len(self.y_train)
        plt.plot(np.arange(n_train), self.y_train, label='Train')
        plt.plot(np.arange(n_train, n_train + len(self.y_test)), self.y_test, label='Test', color='black')
        for name, res in self.best_models:
            plt.plot(np.arange(n_train, n_train + len(self.y_test)), res['y_fore'], label=f'{name}')
        plt.legend()
        plt.title('Forecasts vs Actuals')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return self

    def report(self):
        print('Time Series Forecasting Report')
        for name, res in sorted(self.results.items(), key=lambda x: x[1]['rmse']):
            print(f"- {name}: RMSE={res['rmse']:.3f}, MAE={res['mae']:.3f}, MAPE={res['mape']:.3f}")
        return self

    def run(self):
        (self.load_series()
         .initialize_models()
         .fit_and_forecast()
         .select_best_models()
         .plot_forecasts()
         .report())
        return True


def main():
    parser = argparse.ArgumentParser(description='Run time series forecasting baseline suite')
    parser.add_argument('--dataset', type=str, default='dataset/air_passengers.csv')
    parser.add_argument('--date_col', type=str, default='Date')
    parser.add_argument('--value_col', type=str, default='Value')
    parser.add_argument('--freq', type=str, default='M')
    args = parser.parse_args()

    TSAnalysis(args.dataset, args.date_col, args.value_col, args.freq).run()


if __name__ == '__main__':
    main()

