#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
from src.model import DeepfakeCNN
from src.data_loader import compute_log_mel
import librosa

def load_feature_or_wav(path, sample_rate=16000, duration=3.0):
    path = os.path.normpath(path)
    if path.endswith(".npy"):
        feat = np.load(path).astype(np.float32)
        return feat
    # else wav
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    target_len = int(duration * sample_rate)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start+target_len]
    feat = compute_log_mel(y, sample_rate)
    return feat

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", help="Path to processed WAV file (3s recommended)")
    group.add_argument("--feature", help="Path to .npy log-mel feature")
    p.add_argument("--device", default="cpu")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    model = DeepfakeCNN(n_mels=64).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    path = args.wav if args.wav else args.feature
    feat = load_feature_or_wav(path)
    xb = torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(xb)
        prob = torch.sigmoid(logits).item()
    label = "FAKE" if prob > 0.5 else "REAL"
    print(f"Prob(fake) = {prob:.4f} -> {label}")

if __name__ == "__main__":
    main()
