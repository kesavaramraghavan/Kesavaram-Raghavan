#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

from models.popularity import PopularityRecommender
from models.user_knn import UserKNNRecommender
from models.item_knn import ItemKNNRecommender


def _prepare_example_dataset(path: str, n_users=50, n_items=100, density=0.05, seed=42):
    np.random.seed(seed)
    num_interactions = int(n_users * n_items * density)
    users = np.random.randint(0, n_users, size=num_interactions)
    items = np.random.randint(0, n_items, size=num_interactions)
    ratings = np.ones(num_interactions)
    df = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def train_test_split_interactions(df: pd.DataFrame, test_ratio=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Simple per-user holdout of last interaction
    df = df.sample(frac=1.0, random_state=42)
    test = df.groupby('user_id').head(1)
    train = df.drop(test.index)
    return train, test


def precision_recall_at_k(model, train: pd.DataFrame, test: pd.DataFrame, k=10) -> Tuple[float, float]:
    # For each user in test, recommend and check hits
    hits, total_recommended, total_relevant = 0, 0, 0
    train_seen = train.groupby('user_id')['item_id'].apply(set)
    for user, group in test.groupby('user_id'):
        seen = list(train_seen.get(user, set()))
        recs = model.recommend(user, k=k, seen_items=seen) if hasattr(model, 'recommend') else []
        relevant = set(group['item_id'])
        hits += len(set(recs) & relevant)
        total_recommended += len(recs)
        total_relevant += len(relevant)
    precision = hits / total_recommended if total_recommended else 0.0
    recall = hits / total_relevant if total_relevant else 0.0
    return precision, recall


class RecAnalysis:
    def __init__(self, dataset_path: str, implicit: bool = True):
        self.dataset_path = dataset_path
        self.implicit = implicit
        self.models = {}
        self.results = {}

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            _prepare_example_dataset(self.dataset_path)
        df = pd.read_csv(self.dataset_path)
        if self.implicit:
            df['rating'] = 1.0
        self.df = df[['user_id', 'item_id', 'rating']]
        self.train, self.test = train_test_split_interactions(self.df)
        return self

    def initialize_models(self):
        self.models = {
            'Popularity': PopularityRecommender(),
            'UserKNN': UserKNNRecommender(k=20),
            'ItemKNN': ItemKNNRecommender(k=50),
        }
        return self

    def fit_and_score(self, k=10):
        for name, model in self.models.items():
            model.fit(self.train)
            prec, rec = precision_recall_at_k(model, self.train, self.test, k=k)
            self.results[name] = {'precision': prec, 'recall': rec}
        return self

    def report(self):
        print('Recommendation Systems Report (Precision/Recall@10)')
        for name, m in sorted(self.results.items(), key=lambda x: x[1]['precision'], reverse=True):
            print(f"- {name}: precision={m['precision']:.3f}, recall={m['recall']:.3f}")
        return self

    def run(self):
        (self.load_data()
         .initialize_models()
         .fit_and_score(k=10)
         .report())
        return True


def main():
    parser = argparse.ArgumentParser(description='Run simple recommendation baselines')
    parser.add_argument('--dataset', type=str, default='dataset/interactions.csv')
    parser.add_argument('--implicit', type=int, default=1)
    args = parser.parse_args()

    RecAnalysis(args.dataset, implicit=bool(args.implicit)).run()


if __name__ == '__main__':
    main()

