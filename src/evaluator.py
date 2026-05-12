import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class UnifiedEvaluator:
    """Rigorous comparison between Default, Optuna, and DHPO (Ours)."""
    
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, X_train, y_train, X_test, y_test, params, label="Default"):
        start_time = time.time()
        
        # Ensure parameters are valid integers for RandomForest
        # Handle potential non-integer values from symbolic regression
        clean_params = {}
        for k, v in params.items():
            try:
                clean_params[k] = max(1, int(round(float(v))))
            except (ValueError, TypeError):
                # Fallback to defaults if parsing fails
                defaults = {'max_depth': 10, 'n_estimators': 100}
                clean_params[k] = defaults.get(k, 1)

        model = RandomForestClassifier(**clean_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        eval_time = time.time() - start_time
        
        # Handle multi-class ROC-AUC
        try:
            if len(np.unique(y_test)) > 2:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            auc = 0.0

        return {
            f"{label}_Acc": accuracy_score(y_test, y_pred),
            f"{label}_F1": f1_score(y_test, y_pred, average='weighted'),
            f"{label}_AUC": auc,
            f"{label}_Inference_Time": eval_time
        }
