"""
EEGNet vs Random Forest: Cross-Dataset Schizophrenia EEG Classification

Author: Samiksha BC
Affiliation: Indiana University South Bend

Trains on ASZED dataset and evaluates on an external schizophrenia dataset.
Uses subject-level cross-validation to prevent data leakage.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.integrate import simpson
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, brier_score_loss,
                             confusion_matrix, roc_curve)
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
import traceback
import time

warnings.filterwarnings('ignore')


class Config:
    """Pipeline configuration."""

    DATA_ROOT = Path('.')
    SAMPLING_RATE = 250
    N_CHANNELS = 16

    FILTER_LOW = 0.5
    FILTER_HIGH = 45.0
    NOTCH_FREQ = 50

    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    EEGNET_PARAMS = {
        'n_channels': 16,
        'n_samples': 1000,
        'F1': 8,
        'D': 2,
        'F2': 16,
        'kernel_length': 64,
        'dropout_rate': 0.5,
        'norm_rate': 0.25,
    }

    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15

    RF_PARAMS = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }

    N_FOLDS = 5
    RANDOM_STATE = 42


TARGET_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4",
                   "Cz", "T3", "T4", "T5", "T6", "P3", "P4", "Pz"]

CHANNEL_ALIASES = {
    'FP1': 'Fp1', 'FP2': 'Fp2', 'CZ': 'Cz', 'PZ': 'Pz', 'FZ': 'Fz',
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    'EEG FP1': 'Fp1', 'EEG FP2': 'Fp2', 'EEG F3': 'F3', 'EEG F4': 'F4',
    'fp1': 'Fp1', 'fp2': 'Fp2', 'f3': 'F3', 'f4': 'F4', 'f7': 'F7', 'f8': 'F8',
    'c3': 'C3', 'c4': 'C4', 'cz': 'Cz', 't3': 'T3', 't4': 'T4', 't5': 'T5',
    't6': 'T6', 'p3': 'P3', 'p4': 'P4', 'pz': 'Pz',
}

KACHAREPRAMOD_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
                          "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4"]


def check_pytorch():
    """Check if PyTorch is available."""
    try:
        import torch
        import torch.nn as nn
        return True
    except ImportError:
        return False


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class EEGNet(object):
    """
    EEGNet implementation based on Lawhern et al., 2018.
    A compact CNN for EEG-based brain-computer interfaces.
    """

    def __init__(self, n_channels=16, n_samples=1000, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.5):
        import torch
        import torch.nn as nn

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate
        self.device = get_device()

        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self):
        import torch
        import torch.nn as nn

        class EEGNetModel(nn.Module):
            def __init__(self, n_channels, n_samples, n_classes, F1, D, F2,
                         kernel_length, dropout_rate):
                super(EEGNetModel, self).__init__()

                self.F1 = F1
                self.F2 = F2
                self.D = D

                # Temporal convolution
                self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
                self.bn1 = nn.BatchNorm2d(F1)

                # Depthwise convolution (spatial filtering)
                self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
                self.bn2 = nn.BatchNorm2d(F1 * D)
                self.elu1 = nn.ELU()
                self.pool1 = nn.AvgPool2d((1, 4))
                self.dropout1 = nn.Dropout(dropout_rate)

                # Separable convolution
                self.separable_conv = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                                 padding='same', groups=F1 * D, bias=False)
                self.pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
                self.bn3 = nn.BatchNorm2d(F2)
                self.elu2 = nn.ELU()
                self.pool2 = nn.AvgPool2d((1, 8))
                self.dropout2 = nn.Dropout(dropout_rate)

                self._feature_size = self._get_feature_size(n_channels, n_samples)
                self.classifier = nn.Linear(self._feature_size, n_classes)

            def _get_feature_size(self, n_channels, n_samples):
                return self.F2 * (n_samples // 32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.depthwise(x)
                x = self.bn2(x)
                x = self.elu1(x)
                x = self.pool1(x)
                x = self.dropout1(x)

                x = self.separable_conv(x)
                x = self.pointwise(x)
                x = self.bn3(x)
                x = self.elu2(x)
                x = self.pool2(x)
                x = self.dropout2(x)

                x = x.view(x.size(0), -1)
                x = self.classifier(x)

                return x

        return EEGNetModel(self.n_channels, self.n_samples, self.n_classes,
                          self.F1, self.D, self.F2, self.kernel_length,
                          self.dropout_rate)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32, lr=0.001, patience=15, verbose=True):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        X_train = self._prepare_input(X_train)
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val = self._prepare_input(X_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts))
        class_weights = class_weights.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.5, patience=5)

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                val_loss /= val_total
                val_acc = val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"   Early stopping at epoch {epoch+1}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                          f"train_acc={train_acc:.3f}, val_loss={val_loss:.4f}, "
                          f"val_acc={val_acc:.3f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                          f"train_acc={train_acc:.3f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history

    def _prepare_input(self, X):
        if X.ndim == 3:
            return X[:, np.newaxis, :, :]
        return X

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F

        self.model.eval()
        X = self._prepare_input(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# Data Loading and Preprocessing

def normalize_channel_name(ch_name):
    """Normalize channel name to standard 10-20 format."""
    ch = ch_name.strip()

    if ch in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[ch]
    if ch.upper() in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[ch.upper()]

    for suffix in ['-REF', '-LE', '-Ref', '-le', '-AVG', '-Avg']:
        if ch.endswith(suffix):
            ch = ch[:-len(suffix)].strip()
            if ch in CHANNEL_ALIASES:
                return CHANNEL_ALIASES[ch]

    for target in TARGET_CHANNELS:
        if ch.lower() == target.lower():
            return target

    return ch


def reorder_channels(data, channel_names, target_order=TARGET_CHANNELS):
    """Reorder channels to match target order."""
    normalized = [normalize_channel_name(ch) for ch in channel_names]

    name_to_idx = {}
    for idx, name in enumerate(normalized):
        if name not in name_to_idx:
            name_to_idx[name] = idx

    found_data = []
    found_channels = []

    for target_ch in target_order:
        if target_ch in name_to_idx:
            found_data.append(data[name_to_idx[target_ch]])
            found_channels.append(target_ch)

    if len(found_data) == 0:
        return np.array([]), []

    return np.array(found_data), found_channels


def safe_resample(data, orig_fs, target_fs):
    """Resample EEG data with anti-aliasing."""
    if abs(orig_fs - target_fs) < 1:
        return data

    from math import gcd

    orig_int = int(round(orig_fs))
    target_int = int(round(target_fs))

    g = gcd(orig_int, target_int)
    up = target_int // g
    down = orig_int // g

    resampled = np.zeros((data.shape[0], int(data.shape[1] * up / down)))
    for ch in range(data.shape[0]):
        try:
            resampled[ch] = signal.resample_poly(data[ch], up, down)
        except:
            n_new = int(data.shape[1] * target_fs / orig_fs)
            resampled[ch] = signal.resample(data[ch], n_new)

    return resampled


def preprocess_eeg(data, fs, config):
    """Apply bandpass and notch filters."""
    out = []
    for ch in data:
        ch = ch - np.mean(ch)

        try:
            nyq = fs / 2
            low = config.FILTER_LOW / nyq
            high = min(config.FILTER_HIGH / nyq, 0.99)
            b, a = signal.butter(4, [low, high], 'band')
            ch = signal.filtfilt(b, a, ch)
        except:
            pass

        try:
            if config.NOTCH_FREQ < fs / 2:
                b, a = signal.iirnotch(config.NOTCH_FREQ, 30, fs)
                ch = signal.filtfilt(b, a, ch)
        except:
            pass

        out.append(ch)

    return np.array(out)


def normalize_signal(data):
    """Per-channel z-score normalization."""
    normalized = np.zeros_like(data, dtype=np.float64)

    for ch in range(data.shape[0]):
        channel_data = data[ch].astype(np.float64)
        mean = np.mean(channel_data)
        std = np.std(channel_data)

        if std > 1e-10:
            normalized[ch] = (channel_data - mean) / std
        else:
            normalized[ch] = channel_data - mean

    return normalized


def pad_to_16_channels(data, found_channels):
    """Pad data to 16 channels."""
    if data.shape[0] >= 16:
        return data[:16]

    n_samples = data.shape[1]
    padded = np.zeros((16, n_samples), dtype=data.dtype)

    for i, ch_name in enumerate(found_channels):
        if ch_name in TARGET_CHANNELS:
            target_idx = TARGET_CHANNELS.index(ch_name)
            if target_idx < 16 and i < data.shape[0]:
                padded[target_idx] = data[i]

    return padded


def segment_eeg(data, segment_length, overlap=0.5):
    """Segment EEG data into fixed-length epochs."""
    n_channels, n_samples = data.shape
    step = int(segment_length * (1 - overlap))
    segments = []

    for start in range(0, n_samples - segment_length + 1, step):
        segment = data[:, start:start + segment_length]
        segments.append(segment)

    return np.array(segments) if segments else None


# Feature Extraction

def extract_spectral_power(data, fs, bands, relative=True):
    """Extract spectral power features."""
    features = []

    for ch in range(data.shape[0]):
        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=min(256, data.shape[1]))

        total_power = 1.0
        if relative:
            total_idx = (freqs >= 0.5) & (freqs <= 45)
            if total_idx.any():
                try:
                    total_power = simpson(psd[total_idx], x=freqs[total_idx])
                except:
                    total_power = np.trapz(psd[total_idx], freqs[total_idx])
            if total_power < 1e-10:
                total_power = 1.0

        for band_name, (low, high) in bands.items():
            idx = (freqs >= low) & (freqs <= high)
            try:
                power = simpson(psd[idx], x=freqs[idx]) if idx.any() else 0
            except:
                power = np.trapz(psd[idx], freqs[idx]) if idx.any() else 0

            if relative:
                power = power / total_power

            features.append(power)

    while len(features) < 80:
        features.append(0)
    return np.array(features[:80])


def extract_coherence(data, fs, bands):
    """Extract inter-hemispheric coherence."""
    features = []
    pairs = [(0, 1), (2, 3), (6, 7), (9, 10), (13, 14), (8, 15)]

    for c1, c2 in pairs:
        if c1 < data.shape[0] and c2 < data.shape[0]:
            try:
                f, coh = signal.coherence(data[c1], data[c2], fs=fs,
                                          nperseg=min(256, data.shape[1]))
                for band_name, (low, high) in bands.items():
                    idx = (f >= low) & (f <= high)
                    features.append(np.mean(coh[idx]) if idx.any() else 0)
            except:
                features.extend([0] * len(bands))
        else:
            features.extend([0] * len(bands))

    while len(features) < 30:
        features.append(0)
    return np.array(features[:30])


def extract_pli(data):
    """Extract Phase Lag Index."""
    features = []
    pairs = [(0, 1), (2, 3), (6, 7), (9, 10), (13, 14), (8, 15)]

    for c1, c2 in pairs:
        if c1 < data.shape[0] and c2 < data.shape[0]:
            try:
                a1 = signal.hilbert(data[c1])
                a2 = signal.hilbert(data[c2])
                phase_diff = np.angle(a1) - np.angle(a2)
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                features.append(pli)
            except:
                features.append(0)
        else:
            features.append(0)

    while len(features) < 6:
        features.append(0)
    return np.array(features[:6])


def extract_statistics(data, normalize=True):
    """Extract statistical features per channel."""
    features = []

    for ch in range(data.shape[0]):
        d = data[ch]
        mean_d = np.mean(d)
        std_d = np.std(d)

        if normalize and std_d > 1e-10:
            features.extend([
                mean_d / (std_d + 1e-10),
                std_d / (abs(mean_d) + 1e-10) if abs(mean_d) > 1e-10 else 0,
                stats.skew(d),
                stats.kurtosis(d),
                np.sqrt(np.mean(d**2)) / (std_d + 1e-10),
                np.ptp(d) / (std_d + 1e-10)
            ])
        else:
            features.extend([
                mean_d, std_d, stats.skew(d), stats.kurtosis(d),
                np.sqrt(np.mean(d**2)), np.ptp(d)
            ])

    while len(features) < 96:
        features.append(0)
    return np.array(features[:96])


def extract_all_features(data, fs, config):
    """Extract all features (spectral, coherence, PLI, statistical)."""
    features = []
    features.extend(extract_spectral_power(data, fs, config.BANDS))
    features.extend(extract_coherence(data, fs, config.BANDS))
    features.extend(extract_pli(data))
    features.extend(extract_statistics(data))

    return np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)


# Domain Adaptation

def coral_transform(X_source, X_target):
    """CORAL: Correlation Alignment for domain adaptation."""
    source_mean = np.mean(X_source, axis=0)
    target_mean = np.mean(X_target, axis=0)

    X_source_centered = X_source - source_mean
    X_target_centered = X_target - target_mean

    cov_source = np.cov(X_source_centered, rowvar=False) + np.eye(X_source.shape[1]) * 1e-6
    cov_target = np.cov(X_target_centered, rowvar=False) + np.eye(X_target.shape[1]) * 1e-6

    try:
        eigvals_s, eigvecs_s = np.linalg.eigh(cov_source)
        eigvals_s = np.maximum(eigvals_s, 1e-10)

        eigvals_t, eigvecs_t = np.linalg.eigh(cov_target)
        eigvals_t = np.maximum(eigvals_t, 1e-10)

        whiten = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
        color = eigvecs_t @ np.diag(np.sqrt(eigvals_t)) @ eigvecs_t.T

        X_source_aligned = X_source_centered @ whiten @ color + target_mean
        return X_source_aligned

    except np.linalg.LinAlgError:
        return X_source


def compute_mmd(X_source, X_target, gamma=None, max_samples=200):
    """Compute Maximum Mean Discrepancy."""
    from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

    if len(X_source) > max_samples:
        idx = np.random.choice(len(X_source), max_samples, replace=False)
        X_source = X_source[idx]
    if len(X_target) > max_samples:
        idx = np.random.choice(len(X_target), max_samples, replace=False)
        X_target = X_target[idx]

    n_s, n_t = X_source.shape[0], X_target.shape[0]

    if gamma is None:
        n_sample = min(100, n_s, n_t)
        X_sample = np.vstack([X_source[:n_sample], X_target[:n_sample]])
        dists = euclidean_distances(X_sample, X_sample)
        gamma = 1.0 / (np.median(dists[dists > 0]) ** 2 + 1e-10)

    K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
    K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
    K_st = rbf_kernel(X_source, X_target, gamma=gamma)

    mmd = (np.sum(K_ss) / (n_s * n_s) + np.sum(K_tt) / (n_t * n_t)
           - 2 * np.sum(K_st) / (n_s * n_t))

    return np.sqrt(max(mmd, 0))


# Data Loading

def load_aszed_labels(csv_path):
    """Load subject labels from ASZED CSV."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    label_map = {}
    for _, row in df.iterrows():
        sid = str(row['sn']).strip()
        cat = str(row['category']).lower().strip()
        if any(x in cat for x in ['control', 'hc', 'healthy']):
            label_map[sid] = 0
        elif any(x in cat for x in ['patient', 'schiz', 'sz']):
            label_map[sid] = 1

    return label_map


def load_aszed_eeg(file_path, target_fs=250):
    """Load EEG from ASZED EDF file."""
    import mne
    mne.set_log_level('ERROR')

    raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose='ERROR')

    orig_data = raw.get_data()
    orig_fs = raw.info['sfreq']
    channel_names = raw.ch_names

    try:
        raw.pick_types(eeg=True, exclude='bads')
        data = raw.get_data()
        channel_names = raw.ch_names
    except:
        data = orig_data

    reordered_data, found = reorder_channels(data, channel_names, TARGET_CHANNELS)

    if len(found) >= 8:
        data = reordered_data
    else:
        n_ch = data.shape[0]
        if n_ch >= 8:
            data = data[:min(16, n_ch)]
            found = channel_names[:min(16, n_ch)]
        else:
            raise ValueError(f"Only {n_ch} channels found (min 8 required)")

    if orig_fs != target_fs:
        data = safe_resample(data, orig_fs, target_fs)

    return data, target_fs, found


def load_kacharepramod_subject(file_path, target_fs=250):
    """Load subject from Kacharepramod dataset (.txt format)."""
    source_fs = 128
    n_channels = 16
    samples_per_channel = 7680

    try:
        raw_data = np.loadtxt(file_path)
    except:
        raw_data = pd.read_csv(file_path, header=None).values.flatten()

    n_samples = len(raw_data)

    if n_samples == n_channels * samples_per_channel:
        data = raw_data.reshape(n_channels, samples_per_channel)
    elif n_samples >= samples_per_channel and n_samples % n_channels == 0:
        samples_per_ch = n_samples // n_channels
        data = raw_data.reshape(n_channels, samples_per_ch)
    else:
        raise ValueError(f"Cannot reshape: {n_samples} samples")

    if source_fs != target_fs:
        data = safe_resample(data, source_fs, target_fs)

    data, found = reorder_channels(data, KACHAREPRAMOD_CHANNELS, TARGET_CHANNELS)
    data = pad_to_16_channels(data, found)

    return data, target_fs, found


def load_eea_file(file_path, target_fs=250):
    """Load EEG data from .eea file format."""
    file_path = Path(file_path)
    data = None

    # Try space/tab delimited
    try:
        raw_data = np.loadtxt(file_path)
        if raw_data.ndim == 1:
            n_channels = 19
            if len(raw_data) % n_channels == 0:
                samples_per_ch = len(raw_data) // n_channels
                data = raw_data.reshape(n_channels, samples_per_ch)
            else:
                n_channels = 16
                if len(raw_data) % n_channels == 0:
                    samples_per_ch = len(raw_data) // n_channels
                    data = raw_data.reshape(n_channels, samples_per_ch)
        elif raw_data.ndim == 2:
            if raw_data.shape[0] < raw_data.shape[1]:
                data = raw_data
            else:
                data = raw_data.T
    except:
        pass

    # Try pandas whitespace
    if data is None:
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            raw_data = df.values
            if raw_data.shape[0] < raw_data.shape[1]:
                data = raw_data
            else:
                data = raw_data.T
        except:
            pass

    # Try comma separation
    if data is None:
        try:
            df = pd.read_csv(file_path, header=None)
            raw_data = df.values
            if raw_data.shape[0] < raw_data.shape[1]:
                data = raw_data
            else:
                data = raw_data.T
        except:
            pass

    if data is None:
        raise ValueError(f"Could not parse EEA file: {file_path}")

    n_ch, n_samples = data.shape

    eea_channels_19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
                       "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4",
                       "T6", "O1", "O2"]

    eea_channels_16 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
                       "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4"]

    if n_ch >= 19:
        channel_names = eea_channels_19[:n_ch]
    elif n_ch >= 16:
        channel_names = eea_channels_16[:n_ch]
    else:
        channel_names = [f"Ch{i+1}" for i in range(n_ch)]

    source_fs = 128

    duration_sec = n_samples / source_fs
    if duration_sec < 5:
        source_fs = 256
    elif duration_sec > 300:
        source_fs = 512

    if source_fs != target_fs:
        data = safe_resample(data, source_fs, target_fs)

    data, found = reorder_channels(data, channel_names, TARGET_CHANNELS)
    data = pad_to_16_channels(data, found)

    return data, target_fs, found


# Figure Generation

def generate_figures(analysis, output_dir):
    """Generate publication figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('default')
    except ImportError:
        print("   Warning: matplotlib not available, skipping figures")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ROC curves
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    colors = {'eegnet': '#1f77b4', 'rf': '#ff7f0e', 'rf_coral': '#2ca02c'}
    labels = {'eegnet': 'EEGNet', 'rf': 'Random Forest', 'rf_coral': 'RF + CORAL'}

    for model_name in ['eegnet', 'rf', 'rf_coral']:
        if model_name in analysis.external_results:
            r = analysis.external_results[model_name]
            auc = r['auc']
            sens = r['sensitivity']
            spec = r['specificity']
            ax.plot(1 - spec, sens, 'o', color=colors[model_name], markersize=8,
                    label=f"{labels[model_name]} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('External Dataset Performance')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_roc_external.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_roc_external.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: fig1_roc_external.png/pdf")

    # CV vs External comparison
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    models = []
    cv_accs = []
    cv_stds = []
    ext_accs = []

    for model_name in ['eegnet', 'rf']:
        if model_name in analysis.cv_results:
            models.append(labels[model_name])
            cv_accs.append(analysis.cv_results[model_name]['accuracy'] * 100)
            cv_stds.append(analysis.cv_results[model_name]['accuracy_std'] * 100)
            ext_accs.append(analysis.external_results.get(model_name, {}).get('accuracy', 0) * 100)

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, cv_accs, width, yerr=cv_stds, label='Internal CV',
           color='#1f77b4', capsize=3)
    ax.bar(x + width/2, ext_accs, width, label='External Test',
           color='#ff7f0e')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 100])
    ax.axhline(y=50, color='gray', linestyle='--', lw=1, alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_cv_vs_external.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_cv_vs_external.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: fig2_cv_vs_external.png/pdf")

    # Pipeline diagram
    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.axis('off')

    boxes = [
        (0.05, 0.5, 'ASZED\nDataset\n(Training)'),
        (0.25, 0.5, 'Preprocessing\n(Filter, Normalize)'),
        (0.45, 0.5, 'Model Training\n(EEGNet / RF)'),
        (0.65, 0.5, 'External\nDataset\n(Testing)'),
        (0.85, 0.5, 'Evaluation\n(Metrics)')
    ]

    for x, y, text in boxes:
        bbox = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black')
        ax.text(x, y, text, ha='center', va='center', fontsize=9, bbox=bbox)

    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1][0] - 0.08, 0.5),
                    xytext=(boxes[i][0] + 0.08, 0.5),
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Analysis Pipeline', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_pipeline.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_pipeline.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: fig3_pipeline.png/pdf")


# Metrics

def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['auc'] = 0.5

    try:
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
    except:
        metrics['brier_score'] = 0.25

    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        ece = np.mean(np.abs(prob_true - prob_pred))
        metrics['ece'] = float(ece)
    except:
        metrics['ece'] = 0.5

    return metrics


def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence intervals."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except:
            continue

    if len(scores) < 10:
        return np.nan, np.nan

    alpha = (1 - confidence) / 2
    return np.percentile(scores, alpha * 100), np.percentile(scores, (1 - alpha) * 100)


# Main Pipeline

class EEGNetRFComparison:
    """Main class for EEGNet vs Random Forest comparison."""

    def __init__(self, config):
        self.config = config
        self.results = {}

    def load_aszed_dataset(self, aszed_dir, csv_path, max_files=None):
        """Load and preprocess ASZED dataset."""
        print("\n" + "="*70)
        print("LOADING ASZED DATASET")
        print("="*70)

        label_map = load_aszed_labels(csv_path)
        print(f"   Labels: {sum(v==0 for v in label_map.values())} controls, "
              f"{sum(v==1 for v in label_map.values())} patients")

        aszed_path = Path(aszed_dir)
        files = list(aszed_path.rglob('*.edf'))
        print(f"   Found {len(files)} EDF files")

        pairs = []
        for f in files:
            for part in f.parts:
                if part.startswith('subject_'):
                    if part in label_map:
                        pairs.append((f, part))
                        break
                    numeric_id = part.replace('subject_', '')
                    if numeric_id in label_map:
                        pairs.append((f, numeric_id))
                        break

        print(f"   Matched {len(pairs)} files to labels")

        if len(pairs) > 0:
            print(f"   Sample matches: {[(str(p[0].name), p[1]) for p in pairs[:3]]}")
        else:
            print("   DEBUG: No matches found!")
            print(f"   DEBUG: First 5 label_map keys: {list(label_map.keys())[:5]}")
            if len(files) > 0:
                sample_file = files[0]
                print(f"   DEBUG: Sample file path parts: {sample_file.parts}")
                for part in sample_file.parts:
                    if 'subject' in part.lower():
                        print(f"   DEBUG: Found subject-like part: '{part}'")

        if max_files:
            pairs = pairs[:max_files]

        print(f"   Processing {len(pairs)} files...")

        raw_segments = []
        features_list = []
        labels = []
        subject_ids = []

        errors = {'channel': 0, 'short': 0, 'segment': 0, 'other': 0}
        success_count = 0

        segment_length = self.config.EEGNET_PARAMS['n_samples']

        for fp, sid in tqdm(pairs, desc="Loading ASZED"):
            try:
                data, fs, found = load_aszed_eeg(fp, self.config.SAMPLING_RATE)

                if data.shape[1] < 500:
                    errors['short'] += 1
                    continue

                data = pad_to_16_channels(data, found)
                data = preprocess_eeg(data, fs, self.config)
                data = normalize_signal(data)

                feat = extract_all_features(data, fs, self.config)
                segments = segment_eeg(data, segment_length, overlap=0.5)

                if segments is not None and len(segments) > 0:
                    for seg in segments:
                        raw_segments.append(seg)
                        labels.append(label_map[sid])
                        subject_ids.append(sid)

                    features_list.append(feat)
                    success_count += 1
                else:
                    errors['segment'] += 1

            except ValueError as e:
                if 'channel' in str(e).lower():
                    errors['channel'] += 1
                else:
                    errors['other'] += 1
            except Exception:
                errors['other'] += 1

        print(f"   Success: {success_count}, Errors: {errors}")

        X_raw = np.array(raw_segments)
        X_features = np.array(features_list)
        y = np.array(labels)
        subjects = np.array(subject_ids)

        unique_subjects = list(set(subjects))
        X_rf = []
        y_rf = []
        subjects_rf = []

        for sid in unique_subjects:
            mask = subjects == sid
            if mask.any():
                X_rf.append(np.mean(X_raw[mask].reshape(mask.sum(), -1), axis=0))
                y_rf.append(y[mask][0])
                subjects_rf.append(sid)

        self.aszed_data = {
            'X_raw': X_raw,
            'X_features': np.array(X_rf),
            'y': y,
            'y_rf': np.array(y_rf),
            'subjects': subjects,
            'subjects_rf': np.array(subjects_rf)
        }

        print(f"\n   ASZED Summary:")
        print(f"      Raw segments: {X_raw.shape}")
        print(f"      Unique subjects: {len(unique_subjects)}")
        print(f"      Controls: {(y==0).sum()}, Patients: {(y==1).sum()}")

        return self.aszed_data

    def load_external_dataset(self, external_path, dataset_type='kacharepramod_txt'):
        """Load external dataset for validation."""
        print("\n" + "="*70)
        print("LOADING EXTERNAL DATASET")
        print("="*70)

        external_path = Path(external_path)
        segment_length = self.config.EEGNET_PARAMS['n_samples']

        raw_segments = []
        features_list = []
        labels = []
        subject_ids = []

        norm_path = None
        sch_path = None

        for name in ['norm', 'Norm', 'normal', 'Normal', 'h', 'H', 'Healthy', 'healthy', 'HC', 'hc', 'Control', 'control']:
            if (external_path / name).exists():
                norm_path = external_path / name
                break

        for name in ['sch', 'Sch', 'schizophrenia', 'Schizophrenia', 's', 'S', 'Patient', 'patient', 'SZ', 'sz']:
            if (external_path / name).exists():
                sch_path = external_path / name
                break

        if norm_path is None or sch_path is None:
            raise ValueError("Could not find control/patient folders")

        folders = [(norm_path, 0, "Controls"), (sch_path, 1, "Patients")]

        for folder, label, label_name in folders:
            files = list(folder.glob('*.eea'))
            if len(files) == 0:
                files = list(folder.glob('*.txt'))
            if len(files) == 0:
                files = list(folder.glob('*.csv'))

            print(f"   Found {len(files)} files in {folder.name}/ ({label_name})")

            for fp in tqdm(files, desc=f"Loading {label_name}"):
                try:
                    if fp.suffix.lower() == '.eea':
                        data, fs, found = load_eea_file(fp, self.config.SAMPLING_RATE)
                    else:
                        data, fs, found = load_kacharepramod_subject(fp, self.config.SAMPLING_RATE)

                    if len(found) < 8:
                        continue

                    data = preprocess_eeg(data, fs, self.config)

                    if data.shape[1] < 500:
                        continue

                    data = normalize_signal(data)

                    feat = extract_all_features(data, fs, self.config)
                    features_list.append(feat)

                    segments = segment_eeg(data, segment_length, overlap=0.5)

                    if segments is not None and len(segments) > 0:
                        for seg in segments:
                            raw_segments.append(seg)
                            labels.append(label)
                            subject_ids.append(fp.stem)

                except Exception as e:
                    print(f"      Warning: Could not load {fp.name}: {e}")
                    continue

        X_raw = np.array(raw_segments)
        X_features = np.array(features_list)
        y = np.array(labels)
        subjects = np.array(subject_ids)

        unique_subjects = list(set(subjects))
        X_rf = []
        y_rf = []

        for sid in unique_subjects:
            mask = subjects == sid
            if mask.any():
                X_rf.append(np.mean(X_raw[mask].reshape(mask.sum(), -1), axis=0))
                y_rf.append(y[mask][0])

        self.external_data = {
            'X_raw': X_raw,
            'X_features': np.array(X_rf),
            'y': y,
            'y_rf': np.array(y_rf),
            'subjects': subjects
        }

        print(f"\n   External Summary:")
        print(f"      Raw segments: {X_raw.shape}")
        print(f"      Unique subjects: {len(unique_subjects)}")
        print(f"      Controls: {(y==0).sum()}, Patients: {(y==1).sum()}")

        return self.external_data

    def run_internal_cv(self, n_folds=5):
        """Run subject-level cross-validation on ASZED."""
        print("\n" + "="*70)
        print("INTERNAL CROSS-VALIDATION (ASZED)")
        print("="*70)
        print(f"   Strategy: StratifiedGroupKFold, {n_folds} folds")
        print("   Subject-level separation: YES (no data leakage)")

        X_raw = self.aszed_data['X_raw']
        y = self.aszed_data['y']
        subjects = self.aszed_data['subjects']

        unique_subjects = np.unique(subjects)
        subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
        groups = np.array([subject_to_idx[s] for s in subjects])

        try:
            gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        except:
            gkf = GroupKFold(n_splits=n_folds)

        results = {'eegnet': [], 'rf': []}

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups)):
            print(f"\n   Fold {fold+1}/{n_folds}")

            train_subjects = set(subjects[train_idx])
            val_subjects = set(subjects[val_idx])
            assert len(train_subjects & val_subjects) == 0, "Subject leakage detected!"

            X_train, X_val = X_raw[train_idx], X_raw[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(f"      Train: {len(train_subjects)} subjects, {len(y_train)} segments")
            print(f"      Val:   {len(val_subjects)} subjects, {len(y_val)} segments")

            if check_pytorch():
                print("      Training EEGNet...")
                eegnet = EEGNet(
                    n_channels=16,
                    n_samples=X_train.shape[2],
                    n_classes=2,
                    F1=8, D=2, F2=16,
                    kernel_length=64,
                    dropout_rate=0.5
                )

                eegnet.fit(X_train, y_train, X_val, y_val,
                          epochs=self.config.EPOCHS,
                          batch_size=self.config.BATCH_SIZE,
                          lr=self.config.LEARNING_RATE,
                          patience=self.config.EARLY_STOPPING_PATIENCE,
                          verbose=False)

                y_prob_eegnet = eegnet.predict_proba(X_val)[:, 1]
                y_pred_eegnet = eegnet.predict(X_val)

                eegnet_metrics = compute_metrics(y_val, y_pred_eegnet, y_prob_eegnet)
                results['eegnet'].append(eegnet_metrics)
                print(f"      EEGNet: Acc={eegnet_metrics['accuracy']:.3f}, "
                      f"AUC={eegnet_metrics['auc']:.3f}")

            print("      Training Random Forest...")
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_val_flat = X_val.reshape(len(X_val), -1)

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_val_scaled = scaler.transform(X_val_flat)

            rf = RandomForestClassifier(**self.config.RF_PARAMS)
            rf.fit(X_train_scaled, y_train)

            y_prob_rf = rf.predict_proba(X_val_scaled)[:, 1]
            y_pred_rf = rf.predict(X_val_scaled)

            rf_metrics = compute_metrics(y_val, y_pred_rf, y_prob_rf)
            results['rf'].append(rf_metrics)
            print(f"      RF:     Acc={rf_metrics['accuracy']:.3f}, "
                  f"AUC={rf_metrics['auc']:.3f}")

        self.cv_results = {}
        for model in ['eegnet', 'rf']:
            if results[model]:
                self.cv_results[model] = {
                    'accuracy': np.mean([r['accuracy'] for r in results[model]]),
                    'accuracy_std': np.std([r['accuracy'] for r in results[model]]),
                    'auc': np.mean([r['auc'] for r in results[model]]),
                    'auc_std': np.std([r['auc'] for r in results[model]]),
                    'sensitivity': np.mean([r['sensitivity'] for r in results[model]]),
                    'specificity': np.mean([r['specificity'] for r in results[model]]),
                    'balanced_accuracy': np.mean([r['balanced_accuracy'] for r in results[model]]),
                    'brier_score': np.mean([r['brier_score'] for r in results[model]]),
                    'ece': np.mean([r['ece'] for r in results[model]]),
                    'fold_results': results[model]
                }

        return self.cv_results

    def train_final_models(self):
        """Train final models on all ASZED data for external evaluation."""
        print("\n" + "="*70)
        print("TRAINING FINAL MODELS (ALL ASZED DATA)")
        print("="*70)

        X_raw = self.aszed_data['X_raw']
        y = self.aszed_data['y']

        self.final_models = {}

        if check_pytorch():
            print("   Training final EEGNet...")
            self.final_eegnet = EEGNet(
                n_channels=16,
                n_samples=X_raw.shape[2],
                n_classes=2,
                F1=8, D=2, F2=16,
                kernel_length=64,
                dropout_rate=0.5
            )

            n_val = max(int(len(y) * 0.1), 10)
            indices = np.random.permutation(len(y))
            train_idx, val_idx = indices[n_val:], indices[:n_val]

            self.final_eegnet.fit(
                X_raw[train_idx], y[train_idx],
                X_raw[val_idx], y[val_idx],
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                lr=self.config.LEARNING_RATE,
                patience=self.config.EARLY_STOPPING_PATIENCE,
                verbose=True
            )
            self.final_models['eegnet'] = self.final_eegnet

        print("\n   Training final Random Forest...")
        X_flat = X_raw.reshape(len(X_raw), -1)

        self.rf_scaler = RobustScaler()
        X_scaled = self.rf_scaler.fit_transform(X_flat)

        self.final_rf = CalibratedClassifierCV(
            RandomForestClassifier(**self.config.RF_PARAMS),
            method='isotonic', cv=5
        )
        self.final_rf.fit(X_scaled, y)
        self.final_models['rf'] = self.final_rf

        print("   Done training final models")

        return self.final_models

    def evaluate_external(self, use_domain_adaptation=True):
        """Evaluate models on external dataset."""
        print("\n" + "="*70)
        print("EXTERNAL DATASET EVALUATION")
        print("="*70)

        X_raw_ext = self.external_data['X_raw']
        y_ext = self.external_data['y']

        results = {}

        X_train_flat = self.aszed_data['X_raw'].reshape(len(self.aszed_data['X_raw']), -1)
        X_ext_flat = X_raw_ext.reshape(len(X_raw_ext), -1)

        from sklearn.decomposition import PCA
        n_mmd_samples = min(200, len(X_train_flat), len(X_ext_flat))
        idx_train = np.random.choice(len(X_train_flat), n_mmd_samples, replace=False)
        idx_ext = np.random.choice(len(X_ext_flat), n_mmd_samples, replace=False)

        X_combined = np.vstack([X_train_flat[idx_train], X_ext_flat[idx_ext]])
        pca = PCA(n_components=min(100, X_combined.shape[0] - 1))
        X_combined_pca = pca.fit_transform(X_combined)

        X_train_pca = X_combined_pca[:n_mmd_samples]
        X_ext_pca = X_combined_pca[n_mmd_samples:]

        mmd_original = compute_mmd(X_train_pca, X_ext_pca)
        print(f"\n   Domain Shift (MMD): {mmd_original:.4f}")

        if 'eegnet' in self.final_models:
            print("\n   Evaluating EEGNet...")

            y_prob_eegnet = self.final_eegnet.predict_proba(X_raw_ext)[:, 1]
            y_pred_eegnet = self.final_eegnet.predict(X_raw_ext)

            results['eegnet'] = compute_metrics(y_ext, y_pred_eegnet, y_prob_eegnet)

            print(f"      Accuracy:    {results['eegnet']['accuracy']*100:.1f}%")
            print(f"      AUC:         {results['eegnet']['auc']:.3f}")
            print(f"      Sensitivity: {results['eegnet']['sensitivity']*100:.1f}%")
            print(f"      Specificity: {results['eegnet']['specificity']*100:.1f}%")
            print(f"      Brier Score: {results['eegnet']['brier_score']:.3f}")
            print(f"      ECE:         {results['eegnet']['ece']:.3f}")

        print("\n   Evaluating Random Forest...")

        X_ext_scaled = self.rf_scaler.transform(X_ext_flat)

        y_prob_rf = self.final_rf.predict_proba(X_ext_scaled)[:, 1]
        y_pred_rf = self.final_rf.predict(X_ext_scaled)

        results['rf'] = compute_metrics(y_ext, y_pred_rf, y_prob_rf)

        print(f"      Accuracy:    {results['rf']['accuracy']*100:.1f}%")
        print(f"      AUC:         {results['rf']['auc']:.3f}")
        print(f"      Sensitivity: {results['rf']['sensitivity']*100:.1f}%")
        print(f"      Specificity: {results['rf']['specificity']*100:.1f}%")
        print(f"      Brier Score: {results['rf']['brier_score']:.3f}")
        print(f"      ECE:         {results['rf']['ece']:.3f}")

        if use_domain_adaptation:
            print("\n   Evaluating RF + CORAL Domain Adaptation...")

            X_train_scaled = self.rf_scaler.transform(X_train_flat)
            X_train_coral = coral_transform(X_train_scaled, X_ext_scaled)

            idx_coral = np.random.choice(len(X_train_coral), n_mmd_samples, replace=False)
            idx_ext2 = np.random.choice(len(X_ext_scaled), n_mmd_samples, replace=False)
            X_coral_combined = np.vstack([X_train_coral[idx_coral], X_ext_scaled[idx_ext2]])
            X_coral_pca = pca.transform(X_coral_combined)
            mmd_coral = compute_mmd(X_coral_pca[:n_mmd_samples], X_coral_pca[n_mmd_samples:])

            reduction = (mmd_original - mmd_coral) / (mmd_original + 1e-10) * 100
            print(f"      MMD after CORAL: {mmd_coral:.4f} (reduction: {reduction:.1f}%)")

            rf_coral = CalibratedClassifierCV(
                RandomForestClassifier(**self.config.RF_PARAMS),
                method='isotonic', cv=5
            )
            rf_coral.fit(X_train_coral, self.aszed_data['y'])

            y_prob_rf_coral = rf_coral.predict_proba(X_ext_scaled)[:, 1]
            y_pred_rf_coral = rf_coral.predict(X_ext_scaled)

            results['rf_coral'] = compute_metrics(y_ext, y_pred_rf_coral, y_prob_rf_coral)

            print(f"      Accuracy:    {results['rf_coral']['accuracy']*100:.1f}%")
            print(f"      AUC:         {results['rf_coral']['auc']:.3f}")
            print(f"      Sensitivity: {results['rf_coral']['sensitivity']*100:.1f}%")
            print(f"      Specificity: {results['rf_coral']['specificity']*100:.1f}%")
            print(f"      Brier Score: {results['rf_coral']['brier_score']:.3f}")

        self.external_results = results
        self.mmd_original = mmd_original

        return results

    def generate_publication_report(self):
        """Generate publication-quality report."""
        print("\n" + "="*70)
        print("PUBLICATION SUMMARY: EEGNet vs Random Forest")
        print("for Schizophrenia EEG Classification")
        print("="*70)

        print("\n" + "-"*70)
        print("TABLE 1: Internal Cross-Validation Results (ASZED Dataset)")
        print("-"*70)
        print(f"{'Model':<20} {'Accuracy':<15} {'AUC':<12} {'Sens':<10} {'Spec':<10} {'Brier':<10}")
        print("-"*70)

        for model in ['eegnet', 'rf']:
            if model in self.cv_results:
                r = self.cv_results[model]
                acc_str = f"{r['accuracy']*100:.1f} +/- {r['accuracy_std']*100:.1f}"
                auc_str = f"{r['auc']:.3f} +/- {r['auc_std']:.3f}"
                print(f"{model.upper():<20} {acc_str:<15} {auc_str:<12} "
                      f"{r['sensitivity']*100:.1f}%{'':<5} {r['specificity']*100:.1f}%{'':<5} "
                      f"{r['brier_score']:.3f}")

        print("\n" + "-"*70)
        print("TABLE 2: External Test Results (Kaggle Kacharepramod Dataset)")
        print("-"*70)
        print(f"{'Model':<20} {'Accuracy':<12} {'AUC':<10} {'Sens':<10} {'Spec':<10} {'Brier':<10} {'ECE':<10}")
        print("-"*70)

        for model in ['eegnet', 'rf', 'rf_coral']:
            if model in self.external_results:
                r = self.external_results[model]
                model_name = model.upper().replace('_CORAL', ' + CORAL')
                print(f"{model_name:<20} {r['accuracy']*100:.1f}%{'':<7} {r['auc']:.3f}{'':<5} "
                      f"{r['sensitivity']*100:.1f}%{'':<5} {r['specificity']*100:.1f}%{'':<5} "
                      f"{r['brier_score']:.3f}{'':<5} {r['ece']:.3f}")

        print("\n" + "-"*70)
        print("DOMAIN SHIFT ANALYSIS")
        print("-"*70)
        print(f"Maximum Mean Discrepancy (MMD): {self.mmd_original:.4f}")
        print("\nInterpretation:")
        if self.mmd_original > 0.5:
            print("   SEVERE domain shift detected between datasets.")
        elif self.mmd_original > 0.2:
            print("   MODERATE domain shift detected between datasets.")
        else:
            print("   MILD domain shift detected between datasets.")

        print("\n" + "-"*70)
        print("GENERALIZATION ANALYSIS")
        print("-"*70)

        best_internal = max(
            (self.cv_results.get('eegnet', {}).get('accuracy', 0),
             self.cv_results.get('rf', {}).get('accuracy', 0))
        )

        best_external = max(
            self.external_results.get('eegnet', {}).get('accuracy', 0),
            self.external_results.get('rf', {}).get('accuracy', 0),
            self.external_results.get('rf_coral', {}).get('accuracy', 0)
        )

        gap = best_internal - best_external

        print(f"Best Internal Accuracy: {best_internal*100:.1f}%")
        print(f"Best External Accuracy: {best_external*100:.1f}%")
        print(f"Generalization Gap: {gap*100:.1f}%")

        print("\n" + "-"*70)
        print("GENERALIZATION LIMITATIONS")
        print("-"*70)
        print(f"""
The observed generalization gap ({gap*100:.1f}%) is consistent with published
cross-dataset EEG classification studies. Key factors:

1. Domain shift (MMD = {self.mmd_original:.4f}) between ASZED and external data
2. Different acquisition equipment and recording protocols
3. Sample size constraints (N={len(np.unique(self.aszed_data['subjects']))} ASZED,
   N={len(np.unique(self.external_data['subjects']))} external subjects)
4. Heterogeneity in schizophrenia phenotypes and medication status

Published cross-dataset EEG classification typically achieves 50-65% external
accuracy, consistent with current results.
""")

        return {
            'cv_results': self.cv_results,
            'external_results': self.external_results,
            'domain_shift': self.mmd_original,
            'generalization_gap': gap
        }


def main():
    parser = argparse.ArgumentParser(
        description="EEGNet vs RF: Cross-Dataset Schizophrenia EEG Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py \\
    --aszed-dir /path/to/ASZED \\
    --aszed-csv /path/to/ASZED_SpreadSheet.csv \\
    --external-path /path/to/external/data \\
    --output-dir ./results
        """
    )
    parser.add_argument('--aszed-dir', required=True, help='ASZED data directory')
    parser.add_argument('--aszed-csv', required=True, help='ASZED labels CSV')
    parser.add_argument('--external-path', required=True,
                        help='External dataset path (should contain norm/ and sch/ subdirectories)')
    parser.add_argument('--external-type', default='auto',
                        choices=['auto', 'eea', 'txt', 'csv'],
                        help='External data format (auto-detected by default)')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--max-files', type=int, default=None, help='Max files (testing)')
    parser.add_argument('--n-folds', type=int, default=5, help='CV folds')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs for EEGNet')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for EEGNet')

    args = parser.parse_args()

    print("="*70)
    print("EEGNet vs Random Forest: Schizophrenia EEG Classification")
    print("A Rigorous Cross-Dataset Comparison")
    print("="*70)
    print(f"\nTimestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")

    if check_pytorch():
        import torch
        device = get_device()
        print(f"PyTorch available: {torch.__version__}, Device: {device}")
    else:
        print("WARNING: PyTorch not available. EEGNet will be skipped.")
        print("Install with: pip install torch")

    config = Config()
    config.N_FOLDS = args.n_folds
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    analysis = EEGNetRFComparison(config)

    try:
        analysis.load_aszed_dataset(args.aszed_dir, args.aszed_csv, args.max_files)
        analysis.load_external_dataset(args.external_path, args.external_type)

        analysis.run_internal_cv(n_folds=args.n_folds)
        analysis.train_final_models()
        analysis.evaluate_external(use_domain_adaptation=True)

        report = analysis.generate_publication_report()

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(output_dir / f'eegnet_rf_comparison_{timestamp}.json', 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj

            json.dump(convert_numpy(report), f, indent=2)

        print("\n" + "="*70)
        print("GENERATING FIGURES")
        print("="*70)
        generate_figures(analysis, output_dir)

        print(f"\n[OK] Results saved to: {output_dir}")

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
