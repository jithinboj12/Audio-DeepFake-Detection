#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader
from src.data_loader import DeepfakeDataset
from src.model import DeepfakeCNN
from src.metrics import compute_auc, eer_from_scores
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features-manifest", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    test_ds = DeepfakeDataset(args.metadata, split="test", features_manifest=args.features_manifest)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = DeepfakeCNN(n_mels=64).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="eval"):
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.tolist())
            ys.extend(yb.numpy().tolist())

    auc = compute_auc(ys, preds)
    eer, fpr_at_eer, thr = eer_from_scores(ys, preds)
    print(f"Test AUC: {auc:.4f}")
    print(f"EER: {eer:.4f} (FPR-at-EER: {fpr_at_eer:.4f}, threshold: {thr:.4f})")

if __name__ == "__main__":
    main()
