import numpy as np
import time
from pysr import PySRRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class ESREngine:
    def __init__(self, config):
        self.config = config

    def check_lipschitz_stability(self, Z_grid, threshold=0.15):
        """Topological Audit: Halts pipeline if landscape is not differentiable."""
        gradients = np.gradient(Z_grid)
        max_grad = np.max([np.abs(g).max() for g in gradients])
        if max_grad > threshold:
            raise ValueError(f"Topological Instability Detected (Grad: {max_grad:.3f}). Gating ESR.")
        print(f" [OK] Topological Audit: STABLE (Max Gradient: {max_grad:.4f})")
        return True

    def map_surrogate_landscape(self, X_clean, y, N, D):
        """Dynamically generates search bounds based on dataset geometry."""
        max_depth_upper = min(30, int(np.log2(N) * 2))
        n_est_upper = min(200, D * 5)
        
        X_g, Y_g = np.meshgrid(
            np.linspace(2, max_depth_upper, 5),
            np.linspace(10, n_est_upper, 5)
        )
        Z_g = np.zeros(X_g.shape)
        
        for r in range(X_g.shape[0]):
            for c in range(X_g.shape[1]):
                m = RandomForestClassifier(
                    max_depth=int(X_g[r,c]), 
                    n_estimators=int(Y_g[r,c]), 
                    random_state=42, n_jobs=-1
                )
                Z_g[r,c] = cross_val_score(m, X_clean, y, cv=3).mean()
        
        self.check_lipschitz_stability(Z_g)
        return X_g, Y_g, Z_g

    def discover_law(self, X_g, Y_g, Z_g):
        start_time = time.time()
        
        # Physical Constraint Guardrail: Accuracy must be in [0, 1]
        # We use a custom loss function in Julia to enforce this.
        # PySR allows passing a custom loss as a string of Julia code.
        custom_loss = """
        function custom_loss(prediction, target)
            # Standard MSE
            mse = (prediction - target)^2
            # Penalty for values outside [0, 1]
            penalty = 0.0
            if prediction < 0.0
                penalty = 100.0 * (0.0 - prediction)^2
            elseif prediction > 1.0
                penalty = 100.0 * (prediction - 1.0)^2
            end
            return mse + penalty
        end
        """

        model = PySRRegressor(
            niterations=self.config.SR_ITERATIONS,
            binary_operators=self.config.SR_OPERATORS,
            model_selection="best",
            deterministic=True,
            random_state=42,
            verbosity=0,
            elementwise_loss=custom_loss,
            procs=0,
            multithreading=False
        )
        model.fit(np.column_stack((X_g.flatten(), Y_g.flatten())), Z_g.flatten())
        return str(model.sympy()), time.time() - start_time, Z_g.max()
