import numpy as np


class NaiveBaseline:
    def __init__(self):
        self.last_value = None
        self.model_name = "Naive"

    def fit(self, y):
        self.last_value = float(y[-1])
        return self

    def forecast(self, steps):
        return np.full(steps, self.last_value)


class SeasonalNaiveBaseline:
    def __init__(self, season_length=12):
        self.season_length = season_length
        self.season = None
        self.model_name = f"SeasonalNaive(L={season_length})"

    def fit(self, y):
        if len(y) < self.season_length:
            self.season = np.array([y[-1]])
        else:
            self.season = np.array(y[-self.season_length:])
        return self

    def forecast(self, steps):
        reps = int(np.ceil(steps / len(self.season)))
        return np.tile(self.season, reps)[:steps]


