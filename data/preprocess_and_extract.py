#!/usr/bin/env python3
"""
Preprocess raw audio and extract log-Mel features for Audio DeepFake Detection.

What this script does:
- Scans data/raw/recordings (or the provided --raw-dir) for WAV files.
- Optionally reads a labels CSV mapping raw files to labels (real/fake) and speaker ids.
- For each file:
    - Loads audio, resamples to sample_rate, converts to mono.
    - Trims or pads to fixed duration (seconds).
    - Normalizes amplitude.
    - Saves standardized WAV into out_processed/{real,fake}/ using the same base name.
    - Computes log-Mel spectrogram and saves a .npy file in out_features/.
- Produces two CSVs:
    - metadata.csv (top-level per-clip metadata)
    - features/manifest.csv (per-feature file manifest)

Example:
python data/preprocess_and_extract.py \
    --raw-dir data/raw/recordings \
    --out-processed data/processed \
    --out-features data/features/logmel \
    --labels-csv data/labels.csv \
    --sample-rate 16000 \
    --duration 3.0

If you don't have raw data, run with --generate-demo to synthesize small demo audio files.
"""
import argparse
import os
import glob
import hashlib
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# --------- Utilities ----------
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def write_wav(path, y, sr):
    # sf.write handles float32 data cleanly
    sf.write(path, y.astype(np.float32), sr, subtype='PCM_16')

def compute_log_mel(y, sr, n_mels=64, n_fft=1024, hop_length=256):
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)
    # Standardize per-file
    log_S = (log_S - np.mean(log_S)) / (np.std(log_S) + 1e-9)
    return log_S.astype(np.float32)

# --------- Main processing ----------
def load_labels(labels_csv):
    """
    Expect CSV: raw_path,label,speaker_id,split
    raw_path may be relative to raw_dir.
    """
    mapping = {}
    if not labels_csv or not os.path.exists(labels_csv):
        return mapping
    with open(labels_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row['raw_path']] = {
                'label': row.get('label', 'real'),
                'speaker_id': row.get('speaker_id', ''),
                'split': row.get('split', '')
            }
    return mapping

def synthesize_demo(raw_dir, n=8, sr=16000):
    """
    Create a handful of synthetic WAVs (sine + simple voice-like envelope) for demos.
    """
    ensure_dir(raw_dir)
    dur = 3.0
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    for i in range(n):
        # create a mixture of sinusoids to simulate simple audio
        freqs = [220 + i*10, 440 + i*5, 660 + (i%3)*20]
        y = sum(0.25*np.sin(2*np.pi*f*t) for f in freqs)
        # apply amplitude envelope
        env = np.linspace(0.01, 1.0, len(t))
        y = y * env * 0.9
        # small noise
        y = y + 0.005 * np.random.randn(len(y))
        fname = os.path.join(raw_dir, f"demo_spk{(i%3)+1}_utt{i:03d}.wav")
        sf.write(fname, y.astype(np.float32), sr)
    print(f"Synthesized {n} demo wavs into {raw_dir}")

def process_file(path, out_processed_dir, out_features_dir, sample_rate, duration, labels_map, raw_dir):
    rel = os.path.relpath(path, raw_dir)
    # Determine label and other metadata
    meta = labels_map.get(rel, {'label': 'real', 'speaker_id': '', 'split': ''})
    label = meta.get('label', 'real')
    speaker_id = meta.get('speaker_id', '')
    split = meta.get('split', '')
    # Load
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    # Normalize peak
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.99
    # Pad or trim
    target_len = int(duration * sample_rate)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        # center-crop to be deterministic
        start = max(0, (len(y) - target_len) // 2)
        y = y[start:start+target_len]

    # Prepare out paths
    base = Path(rel).stem
    # Append file variant if label is fake and filename already exists? Keep simple: use same base
    out_label_dir = os.path.join(out_processed_dir, label)
    ensure_dir(out_label_dir)
    out_wav_path = os.path.join(out_label_dir, f"{base}.wav")
    write_wav(out_wav_path, y, sample_rate)

    # Compute features
    feat = compute_log_mel(y, sample_rate)
    ensure_dir(out_features_dir)
    out_feat_path = os.path.join(out_features_dir, f"{base}.npy")
    np.save(out_feat_path, feat)

    # compute md5 of processed wav
    md5 = md5_file(out_wav_path)

    return {
        "filename": out_wav_path.replace(os.sep, '/'),
        "label": label,
        "split": split,
        "speaker_id": speaker_id,
        "duration_sec": f"{duration:.2f}",
        "sample_rate": sample_rate,
        "md5": md5,
        "original_path": path.replace(os.sep, '/'),
        "notes": ""
    }, {
        "feature_file": out_feat_path.replace(os.sep, '/'),
        "audio_file": out_wav_path.replace(os.sep, '/'),
        "n_frames": feat.shape[1],
        "n_mels": feat.shape[0],
        "feature_dtype": str(feat.dtype),
        "shape": str(feat.shape),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

def main():
    p = argparse.ArgumentParser(description="Preprocess raw audio and extract log-mel features.")
    p.add_argument("--raw-dir", required=True, help="Directory containing raw WAVs (scanned recursively).")
    p.add_argument("--out-processed", default="data/processed", help="Where to write standardized WAVs (real/fake).")
    p.add_argument("--out-features", default="data/features/logmel", help="Where to write .npy feature files.")
    p.add_argument("--labels-csv", default="", help="Optional CSV mapping raw_path to label (raw_path,label,speaker_id,split).")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--generate-demo", action="store_true", help="If set and raw dir empty, synthesize demo WAVs.")
    args = p.parse_args()

    raw_dir = args.raw_dir
    ensure_dir(raw_dir)

    # If raw dir is empty and user wants demo, synthesize a few files
    wavs = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)
    if len(wavs) == 0 and args.generate_demo:
        synthesize_demo(raw_dir, n=8, sr=args.sample_rate)
        wavs = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)

    if len(wavs) == 0:
        print(f"No WAV files found in {raw_dir}. Exiting.")
        return

    labels_map = load_labels(args.labels_csv)

    metadata_rows = []
    feat_manifest_rows = []

    for path in sorted(wavs):
        try:
            meta_row, feat_row = process_file(path, args.out_processed, args.out_features, args.sample_rate, args.duration, labels_map, raw_dir)
            metadata_rows.append(meta_row)
            feat_manifest_rows.append(feat_row)
            print(f"Processed {path} -> {meta_row['filename']}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    # Write metadata.csv
    meta_csv_path = os.path.join(os.path.dirname(args.out_processed), "metadata.csv")
    ensure_dir(os.path.dirname(meta_csv_path))
    with open(meta_csv_path, "w", newline='', encoding='utf-8') as f:
        fieldnames = ["filename","label","split","speaker_id","duration_sec","sample_rate","md5","original_path","notes"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in metadata_rows:
            w.writerow(r)
    print(f"Wrote metadata to {meta_csv_path}")

    # Write feature manifest
    feat_manifest_path = os.path.join(os.path.dirname(args.out_features), "manifest.csv")
    ensure_dir(os.path.dirname(feat_manifest_path))
    with open(feat_manifest_path, "w", newline='', encoding='utf-8') as f:
        fieldnames = ["feature_file","audio_file","n_frames","n_mels","feature_dtype","shape","created_at"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in feat_manifest_rows:
            w.writerow(r)
    print(f"Wrote feature manifest to {feat_manifest_path}")

if __name__ == "__main__":
    main()
