import numpy as np
import pandas as pd


class PopularityRecommender:
    def __init__(self):
        self.item_popularity = None
        self.model_name = "Popularity"

    def fit(self, interactions: pd.DataFrame):
        # interactions: columns [user_id, item_id, rating]
        self.item_popularity = interactions.groupby('item_id')['rating'].sum().sort_values(ascending=False)
        return self

    def recommend(self, user_id, k=10, seen_items=None):
        seen = set(seen_items or [])
        recs = [i for i in self.item_popularity.index if i not in seen]
        return recs[:k]


