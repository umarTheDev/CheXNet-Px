"""
CheXNet Model Training (Binary Classification)
----------------------------------------------
This script handles the fine-tuning of the adapted DenseNet121 model.
It supports both single-split training and K-fold cross-validation.

Key Features:
- Differential Learning Rates: Backbone is trained at 0.1x the rate of the head.
- Inverse Frequency Weighting: Handles class imbalance (Positive vs. Negative).
- Automated Hardware Detection (MPS/CPU).
- Early Stopping & Plateau Learning Rate Scheduling.
- Real-time Terminal Progress Bars.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from read_data import ChestXrayDataSet, get_device
from model import load_model

# --- Training Hyperparameters ---
DATA_DIR       = './data/images'
BATCH_SIZE     = 16   # Balanced for Apple Silicon memory
LR             = 1e-4 # Base learning rate for the classifier head
MAX_EPOCHS     = 50
PATIENCE       = 5    # Early stopping patience
LR_FACTOR      = 0.1  # Decay factor for ReduceLROnPlateau

# --- Helper Functions ---

def prompt_training_mode():
    """
    Interactively prompts the user for training configuration.
    
    Returns:
        int: Number of folds (1 for single-split, >1 for k-fold).
    """
    print("\n" + "="*60)
    print(f"{'MODEL TRAINING — CONFIGURATION':^60}")
    print("="*60)
    print("1. Single-split training (70% train, 15% val, 15% test)")
    print("2. K-fold cross-validation")
    
    while True:
        choice = input("\nChoose training mode (1 or 2): ").strip()
        if choice == '1':
            return 1
        elif choice == '2':
            while True:
                try:
                    k = int(input("Enter number of folds (default 5): ").strip() or '5')
                    if k < 2:
                        print("ERROR: Folds must be >= 2.")
                        continue
                    return k
                except ValueError:
                    print("ERROR: Please enter a valid integer.")
        else:
            print("ERROR: Please enter 1 or 2.")

def compute_class_weight(train_list_path):
    """
    Calculates the positive class weight for loss balancing.
    Formula: pos_weight = total_samples / (2 * positive_samples)
    """
    pneumonia_count = 0
    total = 0
    with open(train_list_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            if len(items) >= 2:
                total += 1
                if int(items[1]) == 1:
                    pneumonia_count += 1
    
    if pneumonia_count == 0:
        return 1.0
    
    return total / (2.0 * pneumonia_count)

def print_progress_bar(iteration, total, prefix='', suffix='', length=30):
    """
    Renders a dynamic progress bar in the terminal.
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
    if iteration == total:
        print()

def print_final_summary(aurocs):
    """
    Prints a formatted summary table of results across all folds.
    """
    print("\n" + "="*40)
    print(f"{'CROSS-VALIDATION SUMMARY':^40}")
    print("="*40)
    print(f"{'Fold':<10} | {'Best AUROC':<15}")
    print("-" * 27)
    for i, auroc in enumerate(aurocs):
        print(f"Fold {i:<7} | {auroc:<15.4f}")
    print("-" * 27)
    print(f"{'Mean':<10} | {np.mean(aurocs):<15.4f}")
    print(f"{'Std Dev':<10} | {np.std(aurocs):<15.4f}")
    print("="*40 + "\n")

# --- Main Training Loop ---

def main():
    """
    Orchestrates the training process based on the selected mode.
    """
    device = get_device()

    # Argument Parsing
    parser = argparse.ArgumentParser(description="Train CheXNet Binary Classifier.")
    parser.add_argument('--mode', type=str, choices=['single', 'kfold'], help="Training mode")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    K_FOLDS = args.folds if args.mode == 'kfold' else (1 if args.mode == 'single' else prompt_training_mode())

    # ── Mode 1: Single Split Training ─────────────────────────────────────────
    if K_FOLDS == 1:
        print(f"→ Mode: Single Split Training")
        train_list = './data/train_list.txt'
        val_list = './data/val_list.txt'
        ckpt_path = 'best_binary_model.pth.tar'

        if not os.path.isfile(train_list):
            print("ERROR: Split files not found. Run build_splits.py first.")
            return

        # Data Setup
        train_loader = DataLoader(ChestXrayDataSet(DATA_DIR, train_list, 'train'), 
                                 batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ChestXrayDataSet(DATA_DIR, val_list, 'val'), 
                               batch_size=BATCH_SIZE, shuffle=False)

        # Loss and Weights
        pos_weight = torch.FloatTensor([compute_class_weight(train_list)]).to(device)
        model, _ = load_model()
        base_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Differential Optimization
        # Backbone (pretrained) uses a lower LR than the new classification head
        backbone_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in model.named_parameters() if 'classifier' in name]
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': LR * 0.1}, 
            {'params': head_params, 'lr': LR}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR, patience=2)

        best_val_auroc = 0.0
        patience_counter = 0

        print(f"→ Starting training ({MAX_EPOCHS} epochs max)...")
        for epoch in range(MAX_EPOCHS):
            # Training Phase
            model.train()
            running_loss = 0.0
            for i, (inp, target) in enumerate(train_loader):
                inp, target = inp.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(inp)
                # Weighted Loss Calculation
                loss = base_criterion(output, target)
                weighted_loss = (loss * torch.where(target == 1, pos_weight, 1.0)).mean()
                weighted_loss.backward()
                optimizer.step()
                running_loss += weighted_loss.item()
                if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                    print_progress_bar(i + 1, len(train_loader), prefix=f'Epoch {epoch+1:02d} [Train]', suffix=f'Loss: {weighted_loss.item():.4f}')

            # Validation Phase
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for i, (inp, target) in enumerate(val_loader):
                    output = model(inp.to(device))
                    val_preds.append(output.cpu())
                    val_targets.append(target.cpu())
                    if (i + 1) % 5 == 0 or (i + 1) == len(val_loader):
                        print_progress_bar(i + 1, len(val_loader), prefix=f'Epoch {epoch+1:02d} [Val]  ', suffix='Evaluating...')
                    
            val_preds, val_targets = torch.cat(val_preds), torch.cat(val_targets)
            val_auroc = roc_auc_score(val_targets.numpy().flatten(), torch.sigmoid(val_preds).numpy().flatten())
            
            print(f"  Result: Loss={running_loss/len(train_loader):.4f} | AUROC={val_auroc:.4f} | LR={optimizer.param_groups[0]['lr']:.1e}")

            # Early Stopping and Checkpointing
            scheduler.step(val_auroc)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                torch.save({'state_dict': model.state_dict(), 'best_auroc': best_val_auroc}, ckpt_path)
                print(f"  → Checkpoint saved (New Best AUROC)")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  → Early stopping reached.")
                    break

        print(f"\n→ [SUCCESS] Training complete. Best AUROC: {best_val_auroc:.4f}")
        return

    # ── Mode 2: K-Fold Loop ───────────────────────────────────────────────────
    all_fold_aurocs = []
    for fold in range(K_FOLDS):
        print(f"\n" + "="*60)
        print(f"{f'TRAINING FOLD {fold + 1} / {K_FOLDS}':^60}")
        print("="*60)

        train_list = f'data/folds/train_fold{fold}.txt'
        test_list = f'data/folds/test_fold{fold}.txt'
        ckpt_path = f'best_binary_fold{fold}.pth.tar'

        if not os.path.isfile(train_list):
            print("ERROR: Fold files missing.")
            return

        # DataLoader Setup
        train_loader = DataLoader(ChestXrayDataSet(DATA_DIR, train_list, 'train'), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(ChestXrayDataSet(DATA_DIR, test_list, 'test'), batch_size=BATCH_SIZE, shuffle=False)

        # Initialization
        pos_weight = torch.FloatTensor([compute_class_weight(train_list)]).to(device)
        model, _ = load_model()
        base_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = torch.optim.Adam([
            {'params': [p for name, p in model.named_parameters() if 'classifier' not in name], 'lr': LR * 0.1},
            {'params': [p for name, p in model.named_parameters() if 'classifier' in name], 'lr': LR}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR, patience=2)

        best_val_auroc = 0.0
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            # Training
            model.train()
            running_loss = 0.0
            for i, (inp, target) in enumerate(train_loader):
                inp, target = inp.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(inp)
                loss = (base_criterion(output, target) * torch.where(target == 1, pos_weight, 1.0)).mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                    print_progress_bar(i + 1, len(train_loader), prefix=f'Epoch {epoch+1:02d} [Train]', suffix=f'Loss: {loss.item():.4f}')

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for i, (inp, target) in enumerate(test_loader):
                    output = model(inp.to(device))
                    val_preds.append(output.cpu())
                    val_targets.append(target.cpu())
                    if (i + 1) % 5 == 0 or (i + 1) == len(test_loader):
                        print_progress_bar(i + 1, len(test_loader), prefix=f'Epoch {epoch+1:02d} [Val]  ', suffix='Evaluating...')
            
            val_preds, val_targets = torch.cat(val_preds), torch.cat(val_targets)
            val_auroc = roc_auc_score(val_targets.numpy().flatten(), torch.sigmoid(val_preds).numpy().flatten())
            
            print(f"  Result: Loss={running_loss/len(train_loader):.4f} | AUROC={val_auroc:.4f}\n")
            scheduler.step(val_auroc)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                torch.save({'state_dict': model.state_dict(), 'best_auroc': best_val_auroc}, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        print(f"→ Fold {fold+1} Complete. Best AUROC: {best_val_auroc:.4f}\n")
        all_fold_aurocs.append(best_val_auroc)

    print_final_summary(all_fold_aurocs)

if __name__ == '__main__':
    main()
