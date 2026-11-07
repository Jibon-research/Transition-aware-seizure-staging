# Temporal Staging of Epileptic Seizure Progression

## Project title
**Alpha-Band EEG Dynamics with a Transition-Aware Convolutional GRU Model (ST-ATMR-Net)**

## Overview
This repository contains a full research workflow for EEG-based seizure-stage classification and clinical interpretation. The project is designed to model how a seizure evolves over time from scalp EEG recordings, with emphasis on temporal transition dynamics, patient-disjoint evaluation, and interpretable model behavior.

The codebase integrates:
- data analysis and signal preparation,
- patient-level preprocessing and leakage audits,
- deep learning model training for seizure-stage classification,
- benchmark evaluation across multiple baselines,
- clinical publication figure generation,
- explainable AI analysis of model decisions.

The goal is not just to build a classifier, but to study seizure progression in a way that is clinically meaningful, reproducible, and publication-ready.

---

## Why this research is important
Seizure monitoring is one of the most clinically demanding tasks in epilepsy care. EEG recordings are long, noisy, highly variable across patients, and often reviewed by neurologists under time pressure. Manual interpretation is labor-intensive, subjective, and affected by inter-rater variability, especially when a seizure evolves through multiple transient stages rather than a single discrete event.

This research was motivated by the need for an objective, reproducible, and clinically interpretable way to detect and stage seizure progression from scalp EEG. In practical clinical settings, the distinction between normal brain activity, pre-ictal transition, seizure onset, inter-ictal variability, and post-ictal recovery is crucial because each state carries different diagnostic and treatment implications. A model that can recognize these transitions automatically can support faster clinical decision-making, improve monitoring workflows, reduce human burden, and enable more consistent disease tracking over time.

The broader motivation is also scientific: epilepsy is not a single-state condition but a dynamic process. A seizure is not simply “present” or “absent”; it unfolds in time. By modeling EEG as a temporal progression problem, we can better understand how brain activity changes before, during, and after ictal events. This is a key reason for focusing on temporal dynamics, transition-aware learning, and clinically grounded evaluation rather than only static classification.

In short, this project exists to bridge the gap between raw physiological recordings and clinically meaningful understanding of seizure evolution.

---

## Research problem and motivation
Epileptic seizures are dynamic events that evolve across multiple stages rather than appearing as a single homogeneous state. In clinical monitoring, clinicians often need to understand whether a recording is in a normal, pre-ictal, ictal, inter-ictal, or post-ictal regime, and how transitions between those states occur over time.

This project tackles that problem using multichannel EEG data and temporal deep learning. The code supports both:
- narrow-band alpha-band analysis (8-13 Hz), and
- broadband analysis (0.5-40 Hz),

with the main proposed architecture being a transition-aware temporal model named ST-ATMR-Net.

The labels used throughout the project correspond to staged clinical regimes, including:
- Normal
- Pre-Ictal
- Ictal
- Inter-Ictal
- Post-Ictal

The preprocessing and evaluation workflows are built to respect patient independence and reduce information leakage between training, validation, and test sets.

---

## Study overview and scientific framing
This study formulates seizure-stage detection as a five-class temporal classification problem from multichannel scalp EEG. Rather than treating seizure activity as a single binary event, the approach models seizure evolution as a sequence of clinically meaningful stages spanning from baseline neural activity to ictal onset, inter-ictal dynamics, and post-ictal recovery.

The central scientific question is whether EEG-based temporal patterns can be used to distinguish not only whether seizures occur, but also how the brain transitions through the seizure continuum over time. In other words, the project asks:

1. Can a model learn the progression of seizure-related neurophysiological states from EEG context windows?
2. Do temporal transition features improve stage discrimination beyond raw signal amplitude alone?
3. Can the model generalize across unseen patients while avoiding subject leakage?
4. Does the learned representation remain clinically interpretable and useful for scientific explanation?

The study is therefore not only about classification accuracy. It is also about clinical interpretability, temporal reasoning, reproducibility, and evidence-based neurological analysis.

The task is framed as a sequence-to-label problem, where a model learns a mapping of the form:

`ŷ_t = f(X_{t-W+1:t}, ΔX_t, Δ²X_t)`

where `X` represents the recent EEG context window, `ΔX` is the first-order temporal derivative, and `Δ²X` captures higher-order transition dynamics. This formulation is central to the project because seizure progression is driven by changes over time, not by isolated instantaneous values.

---

## Study design and data strategy
The project is designed around a clinically grounded experimental pipeline:

- Data source: multichannel scalp EEG recordings from patient cohorts with seizure-related annotations.
- Sensor setup: 23-channel clinical montage, consistent with EEG acquisition from scalp recordings.
- Label space: five clinically interpretable stages, including normal, pre-ictal, ictal, inter-ictal, and post-ictal states.
- Temporal modeling: sliding windows over continuous recordings to preserve sequential structure and transition context.
- Frequency analysis: evaluation under both alpha-band (8-13 Hz) and broadband (0.5-40 Hz) inputs.
- Validation: strict patient-grouped splits so that no patient appears in both train and test sets.
- Evaluation emphasis: not only final accuracy, but also balanced accuracy, per-class recall, macro-F1, calibration quality, latency, and clinical consistency.

This combination is essential for building a study that is defensible in a clinical and scientific publication setting. The project explicitly avoids the common weakness of random epoch-level splitting, which can introduce patient identity leakage and over-optimistic performance estimates.

---

## Why the study matters clinically
This research is motivated by the fact that epilepsy care depends on understanding the timing and progression of abnormal brain activity. Neurologists frequently review long EEG recordings to determine not only whether a seizure occurred, but also what stage the patient is in, how quickly the event evolves, and how the brain recovers afterward.

In real clinical settings, early and accurate identification of stage transitions can support:
- faster review of long recordings,
- better monitoring of seizure onset and progression,
- more consistent interpretation across different clinicians,
- objective support for retrospective analysis,
- improved understanding of epileptiform dynamics in patient-specific trajectories.

By modeling the full seizure continuum, rather than only a binary seizure/no-seizure decision, the project aims to move closer to clinically meaningful EEG analysis.

---

## Research contributions of the repository
This project contributes to seizure-stage research in several ways:

- Transition-aware sequence modeling for EEG-based staging
- Patient-disjoint validation to reduce leakage and improve generalizability
- Comparative benchmarking across multiple deep learning baselines
- Alpha-band versus broadband analysis for understanding spectral sensitivity
- Multi-seed stability and ablation studies to assess model robustness
- Explainable AI interpretation for visualizing how the model detects each stage
- Publication-style figure generation for scientific communication

The overall aim is to create a reproducible research framework that can support a journal-quality study in EEG-based epilepsy monitoring.

---

## Core idea of the pipeline
The project follows a standard reproducible scientific workflow:

1. Analyze EEG signals and clinical context
2. Preprocess raw patient data into structured temporal sequences
3. Split recordings by patient rather than by epochs
4. Train multiple models on alpha and broadband data
5. Evaluate performance fairly and compare baselines
6. Generate publication-quality figures and tables
7. Use explainable AI to inspect what the model is detecting

This design makes the repository useful both for research reproduction and for extending the experimental setup to new datasets or architectures.

---

## Technical structure
The repo is organized into six scientific modules plus reusable PyTorch utilities.

```text
02 Code/
├── README.md                              # Project overview and workflow guide
├── run_pipeline.py                        # Master execution harness for all stages
├── checkpoints/                           # Trained model weights (.pt)
│   ├── Alpha_Band/
│   ├── BroadBand/
│   └── Ablation_Studies/
├── src/                                   # Reusable modeling and data utilities
│   ├── paths.py                           # Cross-platform project path resolution
│   ├── dataset.py                         # Sequence construction, normalization, temporal derivatives
│   ├── engine.py                          # Training loops, validation, scheduler logic
│   ├── losses.py                          # Class-weighting and focal-style losses
│   ├── metrics.py                         # Evaluation metrics and plotting utilities
│   ├── simulation.py                      # Clinical timeline and sequence simulation logic
│   ├── utils.py                           # Figures, radar charts, styling, reporting utilities
│   └── models/                            # Deep learning architectures and baselines
├── 01_Data_Analysis/                      # Clinical signal analysis and publication figures
├── 02_Data_Preprocessing/                 # Patient-grouped splits and leakage audits
├── 03_Model_Training_Baselines/            # Training and hyperparameter tuning pipeline
├── 04_Evaluations/                        # Benchmarking, validation, and figure generation
├── 05_XAI_Analysis/                       # Explainable AI and saliency visualizations
├── 06_Notebooks/                          # Interactive notebook analysis
├── archive/                               # Old or superseded experimental scripts
└── LICENSE
```

---

## Main modeling concepts
### 1. EEG sequence modeling
The dataset is built from continuous EEG recordings transformed into sliding sequences. In `src/dataset.py`, each input sample is represented as a temporal window containing:
- context features,
- first-order temporal derivatives,
- second-order temporal derivatives,
- associated class labels.

This is important because seizure progression is not only a function of instantaneous amplitude, but also how signal features evolve over time.

### 2. Causal and leakage-aware preprocessing
The project explicitly separates model fitting from evaluation. In the dataset pipeline, normalization can be performed in ways that respect clinical constraints:
- training-fitted normalization,
- causal baseline calibration,
- session-level normalization.

These options are included to reason carefully about data leakage and to support clinically realistic evaluation.

### 3. Transition-aware architecture
The proposed model, ST-ATMR-Net, is designed around temporal transition awareness and multi-path feature extraction. The codebase compares it against commonly used baselines such as:
- Standard LSTM
- BiLSTM
- TriLSTM
- TCN-1
- TCN-2

The model family is evaluated under both alpha-band and broadband inputs, and also across ablation settings where key components (velocity, acceleration, gating, GRU, attention) are removed or toggled.

### 4. Multi-seed evaluation and ablations
The training workflow includes multi-seed robustness checks and ablation studies. This is important in clinical EEG work because small changes in initialization, class balance, or feature encodings can affect performance dramatically.

---

## Data and experimental setup
The project expects a dataset directory outside the code folder, organized under a parent project structure similar to:

```text
Parent project/
├── 01 Manuscript Preparation/
├── 02 Code/
├── 03 Dataset/
├── 04 Figures/
├── 05 CSV_and_TXT/
└── ...
```

The project resolves paths dynamically via `src/paths.py` so the code can run across machines without hardcoded absolute paths.

The EEG setting assumes multichannel scalp recordings with a 10-20 montage style, using 23 channels and multiple time windows. The dataset is structured around patient-wise partitions to prevent subject overlap between splits.

---

## Reproducible execution workflow
The main entry point is `run_pipeline.py`, which coordinates the scientific pipeline in stages.

### List available stages
```bash
python run_pipeline.py --list
```

### Validate trained checkpoints
```bash
python run_pipeline.py --stage verify
```
This stage checks model integrity and confirms that checkpoint files can be loaded and run through forward passes.

### Stage 1: Data Analysis
```bash
python run_pipeline.py --stage analysis
```
This reproduces the clinical analysis plots such as:
- figure 1: clinical timeline and signal overview,
- figure 2: broadband-to-alpha decomposition,
- figure 5: scalp electrode topography and alpha-band context,
- patient montage visualizations.

### Stage 2: Data Preprocessing
```bash
python run_pipeline.py --stage preprocessing
```
This creates patient-grouped data splits and performs a leakage audit to ensure the train/validation/test partitions are clinically valid.

### Stage 3: Model Training and Benchmarking
The training modules are stored in `03_Model_Training_Baselines/`:

```bash
python 03_Model_Training_Baselines/tune_hyperparameters.py
python 03_Model_Training_Baselines/run_experiments_alpha_band.py
python 03_Model_Training_Baselines/run_experiments_broadband.py
python 03_Model_Training_Baselines/run_multiseed_and_ablation_study.py
```

These scripts train and compare several models under multiple conditions:
- alpha band (8-13 Hz)
- broadband (0.5-40 Hz)
- hyperparameter search
- multi-seed stability analysis
- architectural ablation studies

### Stage 4: Evaluation and publication figures
```bash
python run_pipeline.py --stage evaluations
```
This includes:
- checkpoint verification,
- model auditing,
- paired comparisons,
- radar plots,
- confusion matrices,
- ROC and calibration plots,
- figures 6, 7, and 8.

### Stage 5: Explainable AI
```bash
python run_pipeline.py --stage xai
```
This stage generates saliency-style and topographic interpretations of the model, showing how stage predictions are associated with spatial and temporal EEG features.

### Full pipeline
```bash
python run_pipeline.py --stage all
```
This runs the full research workflow in sequence from analysis to XAI output.

---

## Direct script usage by module

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

### 3) Model training and baselines (`03_Model_Training_Baselines`)
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
```

### 5) XAI Analysis (`05_XAI_Analysis`)
```bash
python 05_XAI_Analysis/generate_authentic_patient_5stage_signals_and_model_xai.py
```

### 6) Notebooks (`06_Notebooks`)
```bash
jupyter notebook 06_Notebooks/01_dataset_analysis_and_montage_visualization.ipynb
```
This notebook helps inspect dataset structure and 23-channel montage rendering interactively.

---

## Expected outputs and artifacts
Running the pipeline generates organized outputs outside the code directory, including:
- trained checkpoints in `checkpoints/`,
- benchmark CSV summaries and metrics JSON files,
- publication figures in the `04 Figures/` structure,
- CSV and text reports in the `05 CSV_and_TXT/` tree,
- diagnostic and XAI visualizations for model interpretation.

The project is designed to support research communication: figures, tables, model comparisons, and validation reports are generated in a structure suitable for papers and internal audits.

---

## Dependencies
The project is based on the Python scientific stack, with major dependencies including:
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-learn

The code is designed to be reproducible and portable across machines, and path configuration is centralized in `src/paths.py`.

---

## Summary
This repository is a full research pipeline for automated temporal seizure staging from EEG. It combines rigorous preprocessing, fair patient-wise evaluation, multi-model benchmarking, and interpretable clinical analysis in one coherent codebase.

The main contributions are:
- clinically organized seizure-stage classification,
- transition-aware temporal modeling,
- patient-disjoint evaluation and leakage audits,
- alpha-band and broadband comparison,
- publication-grade figure generation,
- XAI-driven clinical interpretation.

In other words, this project is not just a model training repo—it is a complete EEG research reproduction and analysis framework for seizure progression.
