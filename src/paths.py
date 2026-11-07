"""
Paths and directory configuration for the project.
Dynamically resolves paths so code works on any machine and with the organized folder layout:
- 01 Manuscript Preparation
- 02 Code
- 03 Dataset
- 04 Figures
- 05 CSV_and_TXT
"""
import os
import sys
from pathlib import Path

# Base directories
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent

# Ensure 02 Code is in sys.path
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MANUSCRIPT_DIR = PROJECT_ROOT / "01 Manuscript Preparation"
DATASET_DIR = PROJECT_ROOT / "03 Dataset"
FIGURES_DIR = PROJECT_ROOT / "04 Figures"
CSV_TXT_DIR = PROJECT_ROOT / "05 CSV_and_TXT"

# Specific subdirectories
CHECKPOINTS_DIR = CODE_DIR / "checkpoints"
ALPHA_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "Alpha_Band"
BROADBAND_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "BroadBand"
ABLATION_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "Ablation_Studies"

ALPHA_DATASET_DIR = DATASET_DIR / "Alpha Band (8-13hz)"
BROADBAND_DATASET_DIR = DATASET_DIR / "All Band (0.5-40hz)"

PUBLICATION_FIGURES_DIR = FIGURES_DIR / "Publication_Figures"
MONTAGES_DIR = FIGURES_DIR / "Patient_Montages"
STAGOGRAMS_DIR = FIGURES_DIR / "Clinical_Stagograms"
DIAGRAMS_AND_ASSETS_DIR = FIGURES_DIR / "Diagrams_and_Assets"

# Organized code subdirectories
DATA_ANALYSIS_DIR = CODE_DIR / "01_Data_Analysis"
DATA_PREPROCESSING_DIR = CODE_DIR / "02_Data_Preprocessing"
MODEL_TRAINING_DIR = CODE_DIR / "03_Model_Training_Baselines"
EVALUATIONS_DIR = CODE_DIR / "04_Evaluations"
XAI_ANALYSIS_DIR = CODE_DIR / "05_XAI_Analysis"
NOTEBOOKS_DIR = CODE_DIR / "06_Notebooks"

PUBLICATION_TABLES_DIR = CSV_TXT_DIR / "Publication_Tables"
DATA_SPLITS_DIR = CSV_TXT_DIR / "Data_Splits_and_Audit"
BENCHMARK_TABLES_DIR = CSV_TXT_DIR / "Model_Benchmarks_and_Ablations"
CLINICAL_SIMULATIONS_DIR = CSV_TXT_DIR / "Clinical_Simulations"
HYPERPARAMETERS_DIR = CSV_TXT_DIR / "Hyperparameters"
MANUSCRIPT_NOTES_DIR = CSV_TXT_DIR / "Manuscript_Drafts_and_Notes"
