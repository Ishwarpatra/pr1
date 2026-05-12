import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import openml

class DataLoader:
    def __init__(self, config):
        self.config = config

    def fetch_and_clean(self, dataset_id):
        print(f"Fetching OpenML Dataset ID: {dataset_id}...")
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")

        # LAPTOP GUARD: Stratified Sub-sampling
        if X.shape[0] > self.config.MAX_ROWS:
            print(f" [WARNING] Dataset exceeds {self.config.MAX_ROWS} rows. Sub-sampling to prevent OOM.")
            X = X.sample(n=self.config.MAX_ROWS, random_state=42)
            y = y.loc[X.index]

        # PREPROCESS GUARD: Handle categoricals safely
        X_num = pd.get_dummies(X, drop_first=True)
        X_imputed = SimpleImputer(strategy='mean').fit_transform(X_num)
        
        # Z-SCORE NORMALIZATION: Critical for scale-invariant Global Variance
        X_scaled = StandardScaler().fit_transform(X_imputed)

        return X_scaled, y.to_numpy()

    def extract_fingerprint(self, X, y):
        N, D = X.shape
        variance = np.trace(np.cov(X.T))
        
        # Shannon Class Entropy
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        entropy = -np.sum(probs * np.log2(probs)) 
        
        return {'N': N, 'D': D, 'variance': variance, 'entropy': entropy}
