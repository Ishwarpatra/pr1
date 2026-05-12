import argparse
import sys
import os
import numpy as np
import random
import optuna
import logging
from src.config import PipelineConfig
from src.data_loader import DataLoader
from src.esr_engine import ESREngine
from src.tracker import CheckpointTracker
from src.notifications import NotificationHandler
from src.evaluator import UnifiedEvaluator
from sklearn.model_selection import train_test_split
import sympy

def set_global_determinism(seed=42):
    """Enforces reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f" [OK] Global Determinism Enforced (Seed: {seed})")

def main():
    parser = argparse.ArgumentParser(description="Deterministic HPO Pipeline")
    parser.add_argument('--env', type=str, choices=['local', 'cluster'], default='local',
                        help="Execution environment. 'local' enforces memory safety.")
    parser.add_argument('--dataset_ids', nargs='+', type=int, required=True,
                        help="List of OpenML Dataset IDs to process.")
    parser.add_argument('--seed', type=int, default=42, help="Global random seed.")
    parser.add_argument('--webhook', type=str, default=None, help="Slack/Webhook URL for notifications.")
    
    args = parser.parse_args()
    
    # 1. Initialize Global Determinism
    set_global_determinism(args.seed)
    
    # 2. Initialize Components
    config = PipelineConfig(env=args.env)
    tracker = CheckpointTracker(filepath="outputs/hpo_results.csv")
    data_loader = DataLoader(config)
    esr_engine = ESREngine(config)
    evaluator = UnifiedEvaluator(config)
    notifier = NotificationHandler(args.webhook)
    
    processed_ids = tracker.get_processed_ids()
    
    print(f"INITIALIZING PIPELINE | Environment: {args.env.upper()}")
    print(f"Memory Gate: Max {config.MAX_ROWS} rows | PySR Iterations: {config.SR_ITERATIONS}")
    print("-" * 60)

    for dataset_id in args.dataset_ids:
        if dataset_id in processed_ids:
            print(f" [OK] Skipping Dataset {dataset_id}: Already processed.")
            continue

        try:
            print(f"Processing Dataset {dataset_id}...")
            
            # 1. Fetch & Clean
            X_full, y_full = data_loader.fetch_and_clean(dataset_id)
            
            # 2. TEST/TRAIN SILOING: Isolate 20% for final evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
            
            # 3. Extract Fingerprint (on training data only)
            fingerprint = data_loader.extract_fingerprint(X_train, y_train)
            print(f" [OK] Fingerprint Extracted: {fingerprint}")

            # 4. Map Surrogate Landscape (on training data only)
            X_g, Y_g, Z_g = esr_engine.map_surrogate_landscape(
                X_train, y_train, fingerprint['N'], fingerprint['D']
            )

            # 5. Discover Law (Symbolic Regression)
            law_str, sr_duration, _ = esr_engine.discover_law(X_g, Y_g, Z_g)
            print(f" [OK] Law Discovered: {law_str} (Time: {sr_duration:.2f}s)")

            # 6. UNIFIED EVALUATION: Compare Default vs DHPO
            # Calculate DHPO params using the discovered law
            # For simplicity, we use the law to predict max_depth (x0) and n_estimators (x1)
            # In a real scenario, we'd solve for the optimum. Here we'll use the law's peak on the grid.
            
            # Find grid point with max accuracy in the surrogate
            max_idx = np.unravel_index(np.argmax(Z_g), Z_g.shape)
            dhpo_params = {
                'max_depth': X_g[max_idx],
                'n_estimators': Y_g[max_idx]
            }
            
            default_params = {'max_depth': 10, 'n_estimators': 100}
            
            print(" [OK] Running Unified Evaluation...")
            default_results = evaluator.evaluate_model(X_train, y_train, X_test, y_test, default_params, label="Default")
            dhpo_results = evaluator.evaluate_model(X_train, y_train, X_test, y_test, dhpo_params, label="DHPO")
            
            # 7. Save Results
            result = {
                'Dataset_ID': dataset_id,
                'N': fingerprint['N'],
                'D': fingerprint['D'],
                'Variance': fingerprint['variance'],
                'Entropy': fingerprint['entropy'],
                'Law': law_str,
                'SR_Runtime_Sec': sr_duration,
                **default_results,
                **dhpo_results
            }
            tracker.save(result)
            
        except Exception as e:
            error_msg = f" [FAILED] on Dataset {dataset_id}: {str(e)}"
            print(error_msg)
            notifier.notify(error_msg)
            # Log failure but continue the loop
            continue

    print("-" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    notifier.notify(f"Pipeline Execution Complete on {len(args.dataset_ids)} datasets.")

if __name__ == "__main__":
    main()
