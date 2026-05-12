import pysr
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class ESREngine:
    def __init__(self, operators: List[str], niterations: int):
        self.operators = operators
        self.niterations = niterations

    def run_symbolic_regression(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        print(f"Running Symbolic Regression with {self.niterations} iterations and operators: {self.operators}")
        
        # PySR expects X as a 2D numpy array and y as a 1D numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        model = pysr.PySRRegressor(
            niterations=self.niterations,
            binary_operators=self.operators,
            unary_operators=[], # Keeping unary operators empty for local env as per directive
            # Other PySR parameters can be added here as needed
            # e.g., populations=10, 
            # procs=0 for single-threaded execution on laptop
            # maxsize=20, 
            # complexity_of_operators={'plus': 1, 'mult': 1, 'div': 1, 'sub': 1}
        )
        
        model.fit(X, y)
        
        # Extract the best equation based on complexity vs. accuracy trade-off
        # This part can be refined based on specific selection criteria
        if model.equations_ is not None and not model.equations_.empty:
            # For simplicity, selecting the equation with the lowest loss
            best_row = model.equations_.iloc[model.equations_['loss'].idxmin()]
            return {
                'equation': best_row['equation'],
                'complexity': best_row['complexity'],
                'loss': best_row['loss']
            }
        else:
            return {
                'equation': 'No equation found',
                'complexity': 0,
                'loss': np.inf
            }
