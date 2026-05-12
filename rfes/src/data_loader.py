import openml
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split # For stratified sub-sampling
from typing import Optional, Tuple

class DataLoader:
    def __init__(self, max_rows: int = 2000):
        self.max_rows = max_rows

    def fetch_and_prep_data(self, dataset_id: int) -> Tuple[np.ndarray, pd.Series]:
        print(f"Fetching OpenML Dataset ID: {dataset_id}...")
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format='dataframe'
        )

        # Apply sub-sampling if dataset exceeds MAX_ROWS
        if X.shape[0] > self.max_rows:
            print(f"Dataset {dataset_id} has {X.shape[0]} rows, exceeding MAX_LOCAL_ROWS ({self.max_rows}). Performing stratified sub-sampling...")
            # Stratified sampling for classification tasks
            if pd.api.types.is_categorical_dtype(y) or y.nunique() < 20: # Heuristic for classification
                _, X_sub, _, y_sub = train_test_split(X, y, train_size=self.max_rows, stratify=y, random_state=42)
            else: # Random sampling for regression tasks
                X_sub = X.sample(n=self.max_rows, random_state=42)
                y_sub = y.loc[X_sub.index]
            X, y = X_sub, y_sub
            print(f"Sub-sampled to {X.shape[0]} rows.")

        X_numeric = pd.get_dummies(X, drop_first=True)
        imputer = SimpleImputer(strategy='mean')
        X_clean = imputer.fit_transform(X_numeric)
        print(f"✓ Dataset {dataset_id} loaded and preprocessed: {X_clean.shape}")
        return X_clean, y

class DatasetFingerprint:
    """Centralized handler for dataset DNA and preprocessing."""
    def __init__(self, X: np.ndarray, y: pd.Series, dataset_id: Optional[int] = None):
        self.dataset_id = dataset_id
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        # Ensure y is a Series for value_counts
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        self.class_imbalance = y.value_counts(normalize=True).min() if len(y.value_counts()) > 1 else 1.0
        self.X = X # X is already preprocessed by DataLoader
        self.y = y

    def summary(self):
        print(f"--- Dataset DNA [{self.dataset_id}] ---")
        print(f"Samples: {self.n_samples} | Features: {self.n_features}")
        print(f"Minority Class Ratio: {self.class_imbalance:.4f}")
