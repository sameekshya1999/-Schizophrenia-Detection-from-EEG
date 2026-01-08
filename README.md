# Schizophrenia EEG Classification: EEGNet vs Random Forest

A cross-dataset comparison of deep learning (EEGNet) and classical machine learning (Random Forest) for schizophrenia detection using EEG signals.

## Author

**Samiksha BC**
Indiana University South Bend

## Overview

This project compares EEGNet and Random Forest classifiers for distinguishing schizophrenia patients from healthy controls using resting-state EEG data. The study emphasizes rigorous cross-dataset validation to assess real-world generalization.

### Key Features

- **Subject-level cross-validation** to prevent data leakage
- **Cross-dataset evaluation** (train on ASZED, test on external data)
- **Domain adaptation** using CORAL (Correlation Alignment)
- **Comprehensive metrics**: Accuracy, AUC, Sensitivity, Specificity, Brier Score, ECE
- **Publication-ready figures** generated automatically

## Requirements

### Python Dependencies

```
numpy
pandas
scipy
scikit-learn
torch
mne
tqdm
matplotlib
joblib
```

### Installation

```bash
pip install numpy pandas scipy scikit-learn torch mne tqdm matplotlib joblib
```

Or using conda:

```bash
conda install numpy pandas scipy scikit-learn pytorch mne tqdm matplotlib joblib -c conda-forge -c pytorch
```

## Dataset Structure

### Training Data (ASZED)
```
ASZED/
  version_1.1/
    node_X/
      subset_X/
        subject_XX/
          session/
            Phase X.edf
ASZED_SpreadSheet.csv  # Contains subject labels (sn, category)
```

### External Test Data
```
external_data/
  norm/           # Healthy controls (.eea, .txt, or .csv files)
  sch/            # Schizophrenia patients (.eea, .txt, or .csv files)
```

## Usage

### Basic Usage

```bash
python main.py \
  --aszed-dir /path/to/ASZED \
  --aszed-csv /path/to/ASZED_SpreadSheet.csv \
  --external-path /path/to/external/data \
  --output-dir ./results
```

### All Options

```bash
python main.py \
  --aszed-dir /path/to/ASZED \
  --aszed-csv /path/to/ASZED_SpreadSheet.csv \
  --external-path /path/to/external/data \
  --external-type auto \
  --output-dir ./results \
  --n-folds 5 \
  --epochs 100 \
  --batch-size 32 \
  --max-files 50  # For testing with subset
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--aszed-dir` | ASZED dataset directory | Required |
| `--aszed-csv` | ASZED labels CSV file | Required |
| `--external-path` | External test dataset path | Required |
| `--external-type` | External data format (auto, eea, txt, csv) | auto |
| `--output-dir` | Output directory for results | ./results |
| `--n-folds` | Number of cross-validation folds | 5 |
| `--epochs` | Maximum training epochs for EEGNet | 100 |
| `--batch-size` | Batch size for EEGNet | 32 |
| `--max-files` | Limit number of files (for testing) | None |

## Output

The pipeline generates:

1. **JSON results file**: `eegnet_rf_comparison_TIMESTAMP.json`
2. **Figures**:
   - `fig1_roc_external.png/pdf` - ROC curves for external evaluation
   - `fig2_cv_vs_external.png/pdf` - Internal CV vs external accuracy comparison
   - `fig3_pipeline.png/pdf` - Analysis pipeline diagram

## Methods

### Preprocessing
- Bandpass filter: 0.5-45 Hz
- Notch filter: 50 Hz
- Z-score normalization per channel
- Resampling to 250 Hz
- Segmentation into 4-second epochs (50% overlap)

### Models

**EEGNet** (Lawhern et al., 2018)
- Temporal convolution + depthwise spatial filtering
- Separable convolutions for efficiency
- Dropout regularization
- Early stopping with validation loss

**Random Forest**
- 500 trees, max depth 15
- Balanced class weights
- Isotonic calibration
- RobustScaler preprocessing

### Domain Adaptation
- CORAL (Correlation Alignment) for distribution matching
- MMD (Maximum Mean Discrepancy) for domain shift quantification



## License

This project is for research and educational purposes.
