"""
Post-Hoc Probability Calibration
-------------------------------
Deep neural networks often produce overconfident or underconfident predictions.
This script implements two popular post-hoc calibration methods to align 
model confidence with actual likelihood:
1. Temperature Scaling: Optimizes a single scalar parameter T to scale logits.
2. Platt Scaling: Fits a logistic regression model on top of sigmoid probabilities.

The script selects the "winner" based on the lowest Brier Score (Mean Squared Error).
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import math
import argparse
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from read_data import ChestXrayDataSet, get_device
from model import DenseNet121Binary

# --- Configuration & Defaults ---
DATA_DIR          = './data/images'
VAL_LIST          = './data/val_list.txt'
CKPT_PATH         = 'best_binary_model.pth.tar'
CALIB_PARAMS_PATH = 'calibration_params.pkl'

# --- Calibration Logic ---

def get_raw_outputs(model, loader, device):
    """
    Collects raw model outputs (sigmoid probabilities) and ground truth labels.
    
    Args:
        model (nn.Module): The trained classifier.
        loader (DataLoader): Validation/Test set loader.
        device (torch.device): Hardware accelerator.
        
    Returns:
        tuple: (flat_probabilities, flat_labels)
    """
    model.eval()
    raw_probs = []
    true_labels = []
    
    print("→ Collecting raw outputs from validation set...")
    with torch.no_grad():
        for i, (inp, target) in enumerate(loader):
            inp = inp.to(device)
            output = model(inp)
            prob = torch.sigmoid(output)
            raw_probs.append(prob.cpu().numpy())
            true_labels.append(target.cpu().numpy())
                
    return np.concatenate(raw_probs).flatten(), np.concatenate(true_labels).flatten()

def get_logits(probs):
    """
    Converts sigmoid probabilities back to log-odds (logits).
    Useful for Temperature Scaling which operates on pre-sigmoid values.
    """
    probs = np.clip(probs, 1e-7, 1 - 1e-7) # Clip to avoid log(0)
    return np.log(probs / (1 - probs))

class TemperatureScaler(nn.Module):
    """
    Simple PyTorch module for Temperature Scaling (p = sigmoid(logit / T)).
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return torch.sigmoid(logits / self.temperature)

def optimize_temperature(logits, labels, device):
    """
    Finds the optimal temperature 'T' by minimizing Negative Log Likelihood 
    (equivalent to BCELoss) on the validation set.
    """
    logits_torch = torch.from_numpy(logits).to(device)
    labels_torch = torch.from_numpy(labels).to(device).float()
    
    scaler = TemperatureScaler().to(device)
    
    # LBFGS is the standard optimizer for single-parameter scaling
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=500)
    criterion = nn.BCELoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_torch), labels_torch)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature.item()

def platt_scale(raw_probs, labels):
    """Fits a Logistic Regression model to the sigmoid probabilities."""
    lr = LogisticRegression()
    lr.fit(raw_probs.reshape(-1, 1), labels)
    return lr

# --- Main Logic ---

def main():
    """
    Orchestrates the calibration process: inference -> T-scaling -> Platt-scaling -> Comparison.
    """
    device = get_device()
    print("\n" + "="*60)
    print(f"{'POST-HOC CALIBRATION':^60}")
    print("="*60)
    
    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Calibrate CheXNet Binary Classifier.")
    parser.add_argument('--mode', type=str, choices=['single', 'kfold'], help="Calibration mode")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds")
    parser.add_argument('--fold-idx', type=int, help="Specific fold index")
    args = parser.parse_args()

    # Determine paths for the selected mode
    ckpt_path = CKPT_PATH
    val_list = VAL_LIST
    calib_params_path = CALIB_PARAMS_PATH

    if args.mode == 'kfold':
        fold_idx = args.fold_idx if args.fold_idx is not None else 0
        ckpt_path = f'best_binary_fold{fold_idx}.pth.tar'
        val_list = f'data/folds/test_fold{fold_idx}.txt'
        calib_params_path = f'calibration_params_fold{fold_idx}.pkl'
        print(f"→ Mode: K-Fold Calibration (Fold {fold_idx})")
    else:
        print(f"→ Mode: Single Split Calibration")

    # 1. Load Model & Data
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        return

    model = DenseNet121Binary()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
        
    val_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=val_list, split='val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

    # 2. Get Raw Predictions
    print("→ Collecting model outputs...")
    raw_probs, true_labels = get_raw_outputs(model, val_loader, device)
    raw_logits = get_logits(raw_probs)

    # 3. Optimize Temperature Scaling
    print("→ Optimizing Temperature Scaling...")
    optimal_t = optimize_temperature(raw_logits, true_labels, device)
    temp_probs = torch.sigmoid(torch.from_numpy(raw_logits) / optimal_t).numpy()
    brier_temp = brier_score_loss(true_labels, temp_probs)

    # 4. Fit Platt Scaling
    print("→ Fitting Platt Scaling...")
    platt_model = platt_scale(raw_probs, true_labels)
    platt_probs = platt_model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    brier_platt = brier_score_loss(true_labels, platt_probs)

    # 5. Result Comparison
    brier_before = brier_score_loss(true_labels, raw_probs)
    
    print("\n" + "="*45)
    print(f"{'CALIBRATION RESULTS':^45}")
    print("="*45)
    print(f"{'Method':<25} | {'Brier Score':<15}")
    print("-" * 45)
    print(f"{'Uncalibrated (Base)':<25} | {brier_before:<15.4f}")
    print(f"{'Temperature Scaling':<25} | {brier_temp:<15.4f}")
    print(f"{'Platt Scaling':<25} | {brier_platt:<15.4f}")
    print("-" * 45)
    
    method = 'temperature' if brier_temp <= brier_platt else 'platt'
    winner_brier = min(brier_temp, brier_platt)
    print(f"WINNER: {method.upper()} Scaling ({brier_before:.4f} → {winner_brier:.4f})")
    print("="*45)

    # 6. Sanity Check: Ensure ranking (AUC) is preserved
    auc_before = roc_auc_score(true_labels, raw_probs)
    final_probs = temp_probs if method == 'temperature' else platt_probs
    auc_after = roc_auc_score(true_labels, final_probs)
    print(f"\nROC-AUC Preservation: {auc_before:.4f} → {auc_after:.4f} [PASSED]" if abs(auc_before-auc_after) < 1e-4 else f"ROC-AUC Preservation: FAILED")

    # 7. Save optimal parameters
    params = {'method': method, 'temperature': optimal_t, 'platt_model': platt_model}
    with open(calib_params_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"→ [SUCCESS] Parameters saved to {calib_params_path}")

if __name__ == "__main__":
    main()
