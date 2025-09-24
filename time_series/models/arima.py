import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.result = None
        self.model_name = f"ARIMA{order}"

    def fit(self, y):
        self.model = ARIMA(y, order=self.order)
        self.result = self.model.fit()
        return self

    def forecast(self, steps):
        return self.result.forecast(steps=steps).values


