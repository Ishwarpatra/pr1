import argparse
import sys
import os
from src.config import PipelineConfig
from src.data_loader import DataLoader, DatasetFingerprint
from src.esr_engine import ESREngine
from src.tracker import Tracker

def main():
    parser = argparse.ArgumentParser(description="Deterministic HPO Pipeline")
    parser.add_argument('--env', type=str, choices=['local', 'cluster'], default='local',
                        help="Execution environment. 'local' enforces memory safety.")
    parser.add_argument('--dataset_ids', nargs='+', type=int, required=True,
                        help="List of OpenML Dataset IDs to process.")
    
    args = parser.parse_args()
    config = PipelineConfig(env=args.env)
    tracker = Tracker(checkpoint_file=os.path.join(os.path.dirname(__file__), "checkpoints.csv"))
    data_loader = DataLoader(max_rows=config.MAX_ROWS)
    esr_engine = ESREngine(operators=config.SR_OPERATORS, niterations=config.SR_ITERATIONS)
    
    print(f"INITIALIZING PIPELINE | Environment: {args.env.upper()}")
    print(f"Memory Gate: Max {config.MAX_ROWS} rows | PySR Iterations: {config.SR_ITERATIONS}")
    print("-" * 60)

    for dataset_id in args.dataset_ids:
        if tracker.is_completed(dataset_id):
            print(f"Skipping Dataset {dataset_id}: Already completed and checkpointed.")
            continue

        try:
            print(f"Processing Dataset {dataset_id}...")
            # 1. Fetch & Sub-sample (data_loader.py)
            X, y = data_loader.fetch_and_prep_data(dataset_id)
            
            # 2. Extract Fingerprint (DatasetFingerprint)
            fingerprint = DatasetFingerprint(X, y, dataset_id)
            fingerprint.summary()

            # 3. GP Surrogate Mapping (Placeholder for now, as per original notebook)
            # This step is not fully implemented in the provided notebook snippet, 
            # so we'll just acknowledge it for now.
            print(f"Placeholder: Performing GP Surrogate Mapping for Dataset {dataset_id}...")

            # 4. Symbolic Regression (esr_engine.py)
            esr_result = esr_engine.run_symbolic_regression(X, y)
            print(f"Symbolic Regression Result for Dataset {dataset_id}: {esr_result}")

            # 5. Save to local CSV (tracker.py)
            tracker.save_checkpoint(dataset_id, status="completed")
            print(f"✓ Dataset {dataset_id} complete and checkpointed.\n")
            
        except Exception as e:
            print(f"⚠ FAILED on Dataset {dataset_id}: {str(e)}")
            tracker.save_checkpoint(dataset_id, status=f"failed: {str(e)}")
            # Do not crash the loop. Log and move to the next.
            continue

if __name__ == "__main__":
    main()
