# Hyperparameter Optimization with Symbolic Regression (rfes)

## Project Overview

This research project discovers mathematical laws governing hyperparameter optimization landscapes using symbolic regression (PySR). It has been refactored from a Jupyter-based environment into a production-ready modular Python architecture to ensure reproducibility, memory safety, and stability across different execution environments (local vs. cluster).

### Key Features:
- **Modular Architecture**: Logic extracted into specialized modules for configuration, data loading, symbolic regression, evaluation, and tracking.
- **Unified Evaluator**: Rigorous comparison between Default and DHPO (Ours) across Accuracy, F1-Score, and ROC-AUC.
- **Test/Train Siloing**: Strict isolation of 20% test split to ensure generalization and prevent data leakage.
- **Memory Safety**: Automatic stratified sub-sampling for large datasets to prevent OOM crashes on local hardware.
- **Environment Gating**: Strict separation of 'local' and 'cluster' constraints for search space and iterations.
- **Robust Checkpointing**: CSV-based state tracking with meta-performance auditing (Win/Loss/Tie).
- **Clean Logging**: Standardized console output with proper dependency handling for long-running processes.

---

## Project Structure

```text
pr1/
├── .gitignore              # Excludes binary junk and execution artifacts
├── Dockerfile              # Containerization for exact environment replication
├── environment.yml         # Conda environment specification
├── main.py                 # CLI entry point for pipeline orchestration
├── requirements.txt        # Strict, version-pinned dependencies
├── src/                    # Modular source code
│   ├── __init__.py
│   ├── config.py           # Environment-specific configurations
│   ├── data_loader.py      # Data fetching and sub-sampling logic
│   ├── esr_engine.py       # PySR symbolic regression engine
│   ├── evaluator.py        # Multi-metric comparative analysis
│   ├── notifications.py    # Webhook notification handler
│   └── tracker.py          # CSV-based checkpointing and meta-summaries
└── tests/                  # Quality assurance infrastructure
    ├── __init__.py
    ├── test_data_loader.py # Unit tests for data loading
    └── test_esr_engine.py  # Unit tests for mathematical core
```

---

## Quick Start

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the pipeline from the terminal using the `main.py` entry point.

**Local Execution (Memory Safe):**
```bash
python main.py --env local --dataset_ids 3 31 1461
```

**Cluster Execution (High Performance):**
```bash
python main.py --env cluster --dataset_ids 3 31 1461
```

---

## Configuration Gating

The pipeline automatically adjusts its behavior based on the `--env` flag:

| Parameter | Local Environment | Cluster Environment |
| :--- | :--- | :--- |
| **MAX_ROWS** | 2,000 (Sub-sampled) | 100,000 |
| **SR_ITERATIONS** | 10 | 40 |
| **SR_OPERATORS** | `+`, `-`, `*`, `/` | `+`, `*`, `-`, `/`, `exp`, `sqrt`, `log` |
| **OPTUNA_TRIALS** | 10 | 100 |

---

## Development & Reproducibility

### Docker
To run the pipeline in a fully isolated container:
```bash
docker build -t rfes-pipeline .
docker run rfes-pipeline --env local --dataset_ids 31
```

### Conda
To replicate the environment using Conda:
```bash
conda env create -f environment.yml
conda activate rfes_env
```

### Testing
Run unit tests to validate the logic:
```bash
python -m unittest discover tests
```

---

## License
This project is licensed under the MIT License. See individual package licenses for third-party dependencies.

**Last Updated:** May 12, 2026  
**Status:** [OK] Production Ready
