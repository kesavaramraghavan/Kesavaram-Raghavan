import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserKNNRecommender:
    def __init__(self, k=20):
        self.k = k
        self.user_item = None
        self.user_index = None
        self.item_index = None
        self.sim = None
        self.model_name = f"UserKNN(k={k})"

    def fit(self, interactions: pd.DataFrame):
        users = interactions['user_id'].astype('category')
        items = interactions['item_id'].astype('category')
        self.user_index = dict(enumerate(users.cat.categories))
        self.item_index = dict(enumerate(items.cat.categories))
        ui = pd.crosstab(users.cat.codes, items.cat.codes, values=interactions['rating'], aggfunc='sum').fillna(0.0)
        self.user_item = ui.values
        self.sim = cosine_similarity(self.user_item)
        return self

    def recommend(self, user_id, k=10):
        # Map user_id to index
        if user_id not in set(self.user_index.values()):
            return []
        inv_user_index = {v: k for k, v in self.user_index.items()}
        u = inv_user_index[user_id]
        # find k nearest users
        neighbors = np.argsort(-self.sim[u])
        neighbors = [idx for idx in neighbors if idx != u][: self.k]
        # score items by neighbor weighted sums
        scores = self.user_item[neighbors].mean(axis=0)
        # remove already interacted items
        seen = set(np.where(self.user_item[u] > 0)[0])
        candidate_idx = [i for i in np.argsort(-scores) if i not in seen]
        items = [self.item_index[i] for i in candidate_idx[:k]]
        return items


