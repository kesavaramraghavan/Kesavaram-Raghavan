import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWintersModel:
    def __init__(self, trend='add', seasonal='add', seasonal_periods=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.result = None
        self.model_name = f"HW({trend},{seasonal},m={seasonal_periods})"

    def fit(self, y):
        self.model = ExponentialSmoothing(y, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods, initialization_method='estimated')
        self.result = self.model.fit()
        return self

    def forecast(self, steps):
        return self.result.forecast(steps=steps).values


