"""
CheXNet Model Evaluation & Benchmarking
---------------------------------------
This script evaluates the trained (and optionally calibrated) model on the test set.
It calculates standard classification metrics and compares them against the 
baseline reproduction targets from Yassen (2025).

Key Metrics:
- ROC-AUC: Area under the Receiver Operating Characteristic curve.
- PR-AUC: Area under the Precision-Recall curve (Average Precision).
- Sensitivity (Recall) & Specificity: Class-specific performance.
- F1-Score: Harmonic mean of precision and recall.
- Confusion Matrix: raw counts of TN, FP, FN, TP.
"""

import os
import torch
import numpy as np
import argparse
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    brier_score_loss
)

from read_data import ChestXrayDataSet, get_device
from model import DenseNet121Binary

# --- Evaluation Configuration ---
DATA_DIR       = './data/images'
CKPT_PATH      = 'best_binary_model.pth.tar'
N_BOOTSTRAP    = 2000 # Number of iterations for confidence interval estimation

# --- Helper Functions ---

def prompt_evaluation_mode():
    """
    Interactively prompts the user for evaluation configuration.
    
    Returns:
        tuple: (mode, fold_idx)
    """
    print("\n" + "="*60)
    print(f"{'MODEL EVALUATION — CONFIGURATION':^60}")
    print("="*60)
    print("1. Single-split evaluation")
    print("2. K-fold evaluation")
    
    while True:
        choice = input("\nChoose evaluation mode (1 or 2): ").strip()
        if choice == '1':
            return 'single', None
        elif choice == '2':
            while True:
                try:
                    fold = int(input("Enter fold index to evaluate (0-4, default 0): ").strip() or '0')
                    return 'kfold', fold
                except ValueError:
                    print("ERROR: Please enter a valid integer.")
        else:
            print("ERROR: Please enter 1 or 2.")

def bootstrap_auroc(gt, pred, n=2000):
    """
    Estimates the 95% Confidence Interval for AUROC using non-parametric bootstrapping.
    """
    rng = np.random.default_rng(42)
    aurocs = []
    for _ in range(n):
        # Resample with replacement
        indices = rng.choice(len(gt), len(gt), replace=True)
        gt_resample = gt[indices]
        pred_resample = pred[indices]
        if len(np.unique(gt_resample)) < 2:
            continue
        aurocs.append(roc_auc_score(gt_resample, pred_resample))
        
    aurocs = np.array(aurocs)
    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)

def find_optimal_threshold(gt, pred):
    """
    Finds the probability threshold that maximizes the F1 score.
    Useful for converting continuous probability estimates into binary decisions.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(gt, (pred >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"Optimal threshold: {best_t:.2f} (Val F1: {best_f1:.4f})")
    return best_t

def get_logits(probs):
    """Utility to convert probabilities back to log-odds."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return np.log(probs / (1 - probs))

# --- Main Evaluation Logic ---

def main():
    """
    Loads the model, generates predictions on the test set, 
    applies calibration, and reports final metrics.
    """
    device = get_device()

    # Argument Parsing
    parser = argparse.ArgumentParser(description="Evaluate CheXNet Binary Classifier.")
    parser.add_argument('--mode', type=str, choices=['single', 'kfold'], help="Evaluation mode")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds")
    parser.add_argument('--fold-idx', type=int, help="Specific fold index")
    args = parser.parse_args()

    mode = args.mode if args.mode else None
    selected_fold = args.fold_idx if args.fold_idx is not None else 0
    if not mode:
        mode, selected_fold = prompt_evaluation_mode()

    # Determine paths based on selection
    if mode == 'kfold':
        test_list = f'data/folds/test_fold{selected_fold}.txt'
        ckpt_path = f'best_binary_fold{selected_fold}.pth.tar'
        print(f"→ Mode: K-Fold Evaluation (Fold {selected_fold})")
    else:
        test_list = 'data/test_list.txt'
        ckpt_path = CKPT_PATH
        print(f"→ Mode: Single Split Evaluation")

    # 1. Model Initialization
    print("→ Initializing Model...")
    model = DenseNet121Binary()
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Model checkpoint {ckpt_path} missing.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # 2. Inference on Test Set
    print("→ Collecting predictions...")
    test_loader = DataLoader(ChestXrayDataSet(DATA_DIR, test_list, 'test'), batch_size=32, shuffle=False)
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inp, target in test_loader:
            output = model(inp.to(device))
            test_preds.append(output.cpu())
            test_targets.append(target.cpu())

    test_probs = torch.sigmoid(torch.cat(test_preds)).numpy().flatten()
    test_targets = torch.cat(test_targets).numpy().flatten()

    # 3. Apply Calibration (if available)
    calib_file = f'calibration_params_fold{selected_fold}.pkl' if mode == 'kfold' else 'calibration_params.pkl'
    if os.path.exists(calib_file):
        print(f"→ Applying Calibration ({calib_file})...")
        with open(calib_file, 'rb') as f:
            calib = pickle.load(f)
        if calib['method'] == 'temperature':
            test_probs = 1 / (1 + np.exp(-get_logits(test_probs) / calib['temperature']))
        elif calib['method'] == 'platt':
            test_probs = calib['platt_model'].predict_proba(test_probs.reshape(-1, 1))[:, 1]
    else:
        print(f"→ [NOTE] No calibration file found. Using raw probabilities.")

    # 4. Metric Calculation
    threshold = find_optimal_threshold(test_targets, test_probs)
    auroc = roc_auc_score(test_targets, test_probs)
    pr_auc = average_precision_score(test_targets, test_probs)
    pred_classes = (test_probs >= threshold).astype(int)
    acc = accuracy_score(test_targets, pred_classes)
    tn, fp, fn, tp = confusion_matrix(test_targets, pred_classes).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(test_targets, pred_classes)

    # 5. Final Reporting
    print("\n" + "="*45)
    print(f"{'FINAL PERFORMANCE SUMMARY':^45}")
    print("="*45)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 45)
    print(f"{'ROC-AUC':<25} | {auroc:<15.4f}")
    print(f"{'PR-AUC':<25} | {pr_auc:<15.4f}")
    print(f"{'Accuracy':<25} | {acc:<15.4f}")
    print(f"{'Sensitivity (Recall)':<25} | {sensitivity:<15.4f}")
    print(f"{'Specificity':<25} | {specificity:<15.4f}")
    print(f"{'F1-score':<25} | {f1:<15.4f}")
    print(f"{'Threshold (Optimal)':<25} | {threshold:<15.2f}")
    print("-" * 45)
    print(f"Confusion Matrix: [TN={tn}, FP={fp} | FN={fn}, TP={tp}]")
    print("="*45)

    # 6. Baseline Comparison (Reproduced from Yassen, 2025)
    print("\n" + "="*55)
    print(f"{'BASELINE COMPARISON (Yassen, 2025)':^55}")
    print("="*55)
    print(f"{'Metric':<20} | {'Current Model':<15} | {'Baseline':<15}")
    print("-" * 55)
    for m, val, base in [("ROC-AUC", auroc, "0.8700"), ("PR-AUC", pr_auc, "0.7200"), 
                         ("Accuracy", acc, "0.9320"), ("Sensitivity", sensitivity, "0.8280"), 
                         ("Specificity", specificity, "0.9460"), ("F1-score", f1, "0.7500")]:
        print(f"{m:<20} | {val:<15.4f} | {base:<15}")
    print("="*55 + "\n")

if __name__ == '__main__':
    main()
