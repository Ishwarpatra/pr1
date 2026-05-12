import os
import pandas as pd

class CheckpointTracker:
    def __init__(self, filepath="outputs/hpo_results.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def get_processed_ids(self):
        """Cross-references existing file to prevent duplicated work."""
        if not os.path.exists(self.filepath):
            return []
        try:
            df = pd.read_csv(self.filepath)
            if 'Dataset_ID' in df.columns:
                return df['Dataset_ID'].tolist()
            return []
        except Exception:
            return []

    def save(self, data_dict):
        """Appends immediately to disk."""
        df_new = pd.DataFrame([data_dict])
        df_new.to_csv(
            self.filepath, 
            mode='a', 
            header=not os.path.exists(self.filepath), 
            index=False
        )
        print(f" [OK] Results for Dataset {data_dict.get('Dataset_ID')} saved to {self.filepath}")
        self._generate_meta_summary()

    def _generate_meta_summary(self):
        """Aggregates performance across all processed datasets."""
        if not os.path.exists(self.filepath):
            return
        
        df = pd.read_csv(self.filepath)
        if 'Default_Acc' not in df.columns or 'DHPO_Acc' not in df.columns:
            return
            
        wins = (df['DHPO_Acc'] > df['Default_Acc']).sum()
        ties = (df['DHPO_Acc'] == df['Default_Acc']).sum()
        losses = (df['DHPO_Acc'] < df['Default_Acc']).sum()
        
        summary_path = os.path.join(os.path.dirname(self.filepath), "meta_summary.txt")
        with open(summary_path, "w") as f:
            f.write("META-PERFORMANCE AUDIT\n")
            f.write("======================\n")
            f.write(f"Total Datasets: {len(df)}\n")
            f.write(f"DHPO Wins: {wins}\n")
            f.write(f"DHPO Ties: {ties}\n")
            f.write(f"DHPO Losses: {losses}\n")
            f.write(f"Win/Tie Rate: {(wins + ties) / len(df) * 100:.2f}%\n")
            f.write(f"Avg DHPO Accuracy: {df['DHPO_Acc'].mean():.4f}\n")
            f.write(f"Avg Default Accuracy: {df['Default_Acc'].mean():.4f}\n")
        print(f" [OK] Meta-summary updated at {summary_path}")
