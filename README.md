# Temporal Staging of Epileptic Seizure Progression
**Alpha-Band EEG Dynamics with a Transition-Aware Convolutional GRU Model (ST-ATMR-Net)**

This directory contains the organized research codebase for preprocessing and analyzing EEG signals, training temporal seizure-stage classifiers, evaluating model architectures, generating publication-grade diagnostic suites, and performing explainable AI (XAI) analysis.

---

## 1. Master Reproduction Harness

You can reproduce individual research phases or run full validation pipelines using the centralized reproduction harness:

```bash
# View all available stages and underlying scripts
python run_pipeline.py --list

# Fast checkpoint validation (verifies all 18 PyTorch checkpoints & forward passes)
python run_pipeline.py --stage verify

# Stage 1: Data Analysis (Figures 1, 2, 5, montages)
python run_pipeline.py --stage analysis

# Stage 2: Data Preprocessing (patient-disjoint partition & leakage audit)
python run_pipeline.py --stage preprocessing

# Stage 4: Model Evaluations (Figures 6, 7, 8, 2x6 ROC/Calibration, 12-Model Matrices)
python run_pipeline.py --stage evaluations

# Stage 5: Explainable AI (XAI) Analysis (5-stage model topomaps & posterior probabilities)
python run_pipeline.py --stage xai

# Run all analysis, preprocessing, evaluation, and XAI stages sequentially
python run_pipeline.py --stage all
```

---

## 2. Organized Directory Structure

The code is strictly structured into 6 logical scientific modules:

```text
02 Code/
├── run_pipeline.py                         # Master CLI pipeline runner
├── checkpoints/                            # Pre-trained PyTorch weights (.pt)
│   ├── Alpha_Band/                         # Alpha Band (8-13 Hz) model checkpoints
│   ├── BroadBand/                          # BroadBand (0.5-40 Hz) model checkpoints
│   └── Ablation_Studies/                   # Architectural component ablation checkpoints
├── src/                                    # Modular PyTorch package
│   ├── paths.py                            # Dynamic cross-platform path resolution
│   ├── dataset.py                          # Causal loaders & sliding window sequence builders
│   ├── engine.py                           # Training, validation, warmup + cosine learning scheduler
│   ├── losses.py                           # Multiclass Focal Loss & smooth weighting
│   ├── metrics.py                          # Latency measurements, confusion matrices, ROC/PR
│   ├── simulation.py                       # Streaming timeline & event transition simulator
│   ├── utils.py                            # Radar charts, learning curves, Excel styling
│   └── models/                             # ST-ATMR-Net, Standard LSTM, BiLSTM, TriLSTM, TCN-1, TCN-2
│
├── 01_Data_Analysis/                       # Raw signal & spectral analysis
│   ├── generate_trustworthy_figure1_physionet_offset.py # Figure 1: Clinical Timeline (EDF, spectrograms)
│   ├── generate_publication_figure2_broadband_to_alpha.py # Figure 2: BroadBand to Alpha Decomposition
│   ├── generate_clean_publication_figure5.py # Figure 5: 10-20 Scalp Electrode Layout & Alpha Topography
│   └── generate_authentic_patient_montages.py # 23-Channel continuous clinical EEG montages
│
├── 02_Data_Preprocessing/                  # Dataset splitting & leakage auditing
│   └── create_patient_grouped_splits.py    # Strict patient-grouped partition & audit report
│
├── 03_Model_Training_Baselines/            # Model training & hyperparameter search
│   ├── tune_hyperparameters.py             # Validation-isolated hyperparameter grid search
│   ├── run_experiments_alpha_band.py       # Full benchmark training for Alpha Band (8-13 Hz)
│   ├── run_experiments_broadband.py        # Full benchmark training for BroadBand (0.5-40 Hz)
│   └── run_multiseed_and_ablation_study.py # 5-Seed stability training & 6 architectural ablations
│
├── 04_Evaluations/                         # Comprehensive evaluation & diagnostic suites
│   ├── verify_and_load_checkpoints.py      # Checkpoint integrity & inference harness
│   ├── verify_all_critical_issues.py       # Paired comparison, latency, transition compliance
│   ├── generate_audited_figures6_and_7.py  # Figures 6 & 7: Multi-Seed Boxplots & Ablation Drop
│   ├── generate_verification_and_paired_comparison_figure.py # Figure 8: Clinical Audit & Paired Comparison
│   ├── generate_all_6_models_comparative_diagnostic_suite.py # 2x3 Comparative diagnostic suite
│   ├── generate_combined_12_model_confusion_matrices.py # 12-Model 2x6 Confusion Matrix master suite
│   ├── generate_combined_12_model_roc_and_calibration_2x6.py # 12-Model 2x6 ROC & Calibration master suite
│   ├── generate_combined_radar_comparison.py # Combined 1x2 Clinical Radar Benchmark
│   ├── generate_publication_diagnostic_figures.py # Alpha Band diagnostic ROC, PR & Calibration
│   └── generate_broadband_publication_diagnostic_figures.py # BroadBand diagnostic ROC, PR & Calibration
│
├── 05_XAI_Analysis/                        # Explainable AI & saliency attribution
│   ├── generate_authentic_patient_5stage_signals_and_model_xai.py # Figure: 5-Stage Signals & XAI Saliency Topomap
│   └── generate_all_recordings_signals_and_model_xai.py # Full-cohort recording-wise XAI detection suite
│
└── 06_Notebooks/                           # Interactive visualization notebooks
    └── 01_dataset_analysis_and_montage_visualization.ipynb # Interactive dataset analysis & montages
```

---

## 3. Direct Script Execution Guide

Every script can be executed either directly from `02 Code/` or from within its own subfolder. Dynamic path resolution guarantees that all checkpoints, datasets, figures, and CSV outputs are correctly located regardless of the working directory:

### 1) Data Analysis (`01_Data_Analysis`)
```bash
python 01_Data_Analysis/generate_trustworthy_figure1_physionet_offset.py
python 01_Data_Analysis/generate_publication_figure2_broadband_to_alpha.py
python 01_Data_Analysis/generate_clean_publication_figure5.py
python 01_Data_Analysis/generate_authentic_patient_montages.py
```

### 2) Data Preprocessing (`02_Data_Preprocessing`)
```bash
python 02_Data_Preprocessing/create_patient_grouped_splits.py
```

### 3) Model Training & Baselines (`03_Model_Training_Baselines`)
```bash
python 03_Model_Training_Baselines/tune_hyperparameters.py
python 03_Model_Training_Baselines/run_experiments_alpha_band.py
python 03_Model_Training_Baselines/run_experiments_broadband.py
python 03_Model_Training_Baselines/run_multiseed_and_ablation_study.py
```

### 4) Evaluations (`04_Evaluations`)
```bash
python 04_Evaluations/verify_and_load_checkpoints.py
python 04_Evaluations/verify_all_critical_issues.py
python 04_Evaluations/generate_audited_figures6_and_7.py
python 04_Evaluations/generate_verification_and_paired_comparison_figure.py
python 04_Evaluations/generate_combined_12_model_confusion_matrices.py
python 04_Evaluations/generate_combined_12_model_roc_and_calibration_2x6.py
python 04_Evaluations/generate_combined_radar_comparison.py
python 04_Evaluations/generate_all_6_models_comparative_diagnostic_suite.py
python 04_Evaluations/generate_publication_diagnostic_figures.py
python 04_Evaluations/generate_broadband_publication_diagnostic_figures.py
```

### 5) XAI Analysis (`05_XAI_Analysis`)
```bash
python 05_XAI_Analysis/generate_authentic_patient_5stage_signals_and_model_xai.py
python 05_XAI_Analysis/generate_all_recordings_signals_and_model_xai.py
```

### 6) Notebooks (`06_Notebooks`)
Launch Jupyter Notebook / JupyterLab:
```bash
jupyter notebook 06_Notebooks/01_dataset_analysis_and_montage_visualization.ipynb
```
The notebook automatically resolves project paths to `03 Dataset` and renders all interactive 23-channel montages.
