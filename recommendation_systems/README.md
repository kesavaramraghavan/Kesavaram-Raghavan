#!/usr/bin/env python3
# Recommendation Systems Learning Project

Simple recommenders with a unified CLI and plots.

## Approaches
- Popularity baseline (`popularity.py`)
- User-based kNN collaborative filtering (`user_knn.py`)
- Item-based kNN collaborative filtering (`item_knn.py`)

## Data Format
CSV with columns: `user_id,item_id,rating` (implicit feedback supported by treating any interaction as rating=1).

## Usage
```
python main.py --dataset dataset/interactions.csv --implicit 1
```

Outputs top-N recommendations and offline metrics (precision@k, recall@k) via simple holdout.


