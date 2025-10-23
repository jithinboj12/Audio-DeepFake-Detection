import os
import csv
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

DEFAULT_SR = 16000
DEFAULT_DURATION = 3.0  # seconds
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

LABEL_MAP = {"real": 0, "fake": 1}

def compute_log_mel(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = (log_S - np.mean(log_S)) / (np.std(log_S) + 1e-9)
    return log_S.astype(np.float32)

def load_npy_feature(path):
    return np.load(path).astype(np.float32)

class DeepfakeDataset(Dataset):
    """
    Dataset that uses metadata.csv. If features_manifest is provided and a feature entry exists,
    the dataset will load precomputed .npy features. Otherwise it will compute log-mel from the WAV.
    """
    def __init__(self, metadata_csv, split="train", features_manifest=None,
                 sample_rate=DEFAULT_SR, duration=DEFAULT_DURATION, n_mels=N_MELS, preload=False):
        self.df = pd.read_csv(metadata_csv)
        # filter by split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows with split={split} in {metadata_csv}")
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.preload = preload

        # load feature manifest mapping if provided
        self.feature_map = {}
        if features_manifest and os.path.exists(features_manifest):
            try:
                fdf = pd.read_csv(features_manifest)
                # map audio_file -> feature_file (normalize paths)
                for _, r in fdf.iterrows():
                    audio = os.path.normpath(r['audio_file'])
                    feat = os.path.normpath(r['feature_file'])
                    self.feature_map[audio] = feat
            except Exception:
                # ignore malformed manifest
                self.feature_map = {}

        # Preload features or waveforms if requested
        if preload:
            self._cache = []
            for _, row in self.df.iterrows():
                feat = self._load_feature_for_row(row)
                self._cache.append((feat, row))
        else:
            self._cache = None

    def __len__(self):
        return len(self.df)

    def _fix_audio_len(self, y):
        target_len = int(self.duration * self.sample_rate)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            # center crop
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]
        return y

    def _load_feature_for_row(self, row):
        audio_path = os.path.normpath(row['filename'])
        # if feature exists, load it
        if audio_path in self.feature_map and os.path.exists(self.feature_map[audio_path]):
            feat = load_npy_feature(self.feature_map[audio_path])
            return feat
        # otherwise compute from wav
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        y = self._fix_audio_len(y)
        feat = compute_log_mel(y, self.sample_rate, n_mels=self.n_mels)
        return feat

    def __getitem__(self, idx):
        if self._cache is not None:
            feat, row = self._cache[idx]
            label = LABEL_MAP.get(row['label'], 0)
        else:
            row = self.df.iloc[idx]
            feat = self._load_feature_for_row(row)
            label = LABEL_MAP.get(row['label'], 0)
        # Convert to tensor shaped (1, n_mels, time)
        x = torch.tensor(feat).unsqueeze(0)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y
