import pandas as pd
import os

class Tracker:
    def __init__(self, checkpoint_file="checkpoints.csv"):
        self.checkpoint_file = checkpoint_file
        self.checkpoints = self._load_checkpoints()

    def _load_checkpoints(self):
        if os.path.exists(self.checkpoint_file):
            return pd.read_csv(self.checkpoint_file)
        return pd.DataFrame(columns=["dataset_id", "status", "timestamp"])

    def save_checkpoint(self, dataset_id: int, status: str = "completed"):
        new_checkpoint = pd.DataFrame([{"dataset_id": dataset_id, "status": status, "timestamp": pd.Timestamp.now()}])
        self.checkpoints = pd.concat([self.checkpoints, new_checkpoint], ignore_index=True)
        self.checkpoints.to_csv(self.checkpoint_file, index=False)
        print(f"✓ Checkpoint saved for Dataset {dataset_id} with status: {status}")

    def is_completed(self, dataset_id: int) -> bool:
        return (self.checkpoints["dataset_id"] == dataset_id).any()

    def get_last_completed_dataset(self) -> Optional[int]:
        completed = self.checkpoints[self.checkpoints["status"] == "completed"]
        if not completed.empty:
            return completed["dataset_id"].iloc[-1]
        return None
