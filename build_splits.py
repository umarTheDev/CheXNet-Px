"""
CheXNet Dataset Preparation (Splitting & Stratification)
--------------------------------------------------------
This script processes the master labels file and generates training, validation,
and testing splits. It supports both a single 70/15/15 split and K-fold
cross-validation.

Key Features:
- Stratified splitting (maintains class distribution).
- Group-aware splitting (ensures same patient images stay in the same split).
- Dataset balancing based on target counts and disease distribution.
"""

import pandas as pd
import numpy as np
import math
import os
import shutil
import argparse
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

# --- Configuration & Path Constants ---

# Master label file location (Space-separated: image_name label1 label2 ... label14)
INPUT_FILE      = 'data/labels.txt'

# Target composition for the final subset
TARGET_PNEUMONIA = 444
TARGET_NORMAL    = math.floor(7.1 * TARGET_PNEUMONIA)  # Maintains approximately 7.1 : 1 ratio

# Reproducibility seed
RANDOM_SEED      = 42

# Original NIH Dataset labels
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
               'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
               'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

PNEUMONIA_IDX = CLASS_NAMES.index('Pneumonia')  # Index 6

# --- Helper Functions ---

def prompt_split_options():
    """
    Interactively prompts the user for dataset splitting configuration.
    
    Returns:
        tuple: (generate_single_split, generate_k_fold, k_folds)
    """
    print("\n" + "="*60)
    print(f"{'DATASET PREPARATION — CONFIGURATION':^60}")
    print("="*60)
    
    single_split = input("Generate single-split files? (y/n) [default: y]: ").strip().lower() or 'y'
    single_split = single_split == 'y'
    
    k_fold = input("Generate k-fold split files? (y/n) [default: y]: ").strip().lower() or 'y'
    k_fold = k_fold == 'y'
    
    k_folds = 5
    if k_fold:
        while True:
            try:
                k_folds = int(input("Enter number of folds (default 5): ").strip() or '5')
                if k_folds < 2:
                    print("ERROR: Number of folds must be >= 2")
                    continue
                break
            except ValueError:
                print("ERROR: Please enter a valid integer")
                
    return single_split, k_fold, k_folds

def write_list(df, path):
    """
    Saves a DataFrame split to a space-separated .txt file.
    Format: [image_name] [binary_label]
    """
    with open(path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{row['image_name']} {row['binary_label']}\n")

def print_stats_table(stats_dict):
    """
    Prints a formatted summary table of the generated dataset splits.
    
    Args:
        stats_dict (dict): Dictionary mapping split names to DataFrames.
    """
    header = f"{'Split':<10} | {'Total':<8} | {'Normal':<8} | {'Pneumonia':<10} | {'Ratio (N:P)':<12}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)
    for name, df in stats_dict.items():
        total = len(df)
        pneumonia = df['binary_label'].sum()
        normal = total - pneumonia
        ratio = f"{(normal / pneumonia):.1f}:1" if pneumonia > 0 else "N/A"
        print(f"{name:<10} | {total:<8} | {normal:<8} | {pneumonia:<10} | {ratio:<12}")
    print(divider + "\n")

# --- Main Processing Logic ---

def main():
    """
    Main logic for loading, filtering, balancing, and splitting the dataset.
    Preserves the complex pooling logic for pneumonia and no-finding images.
    """
    parser = argparse.ArgumentParser(description="Split CheXNet dataset into Train/Val/Test.")
    parser.add_argument('--mode', type=str, choices=['single', 'kfold', 'both'], help="Split mode")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    # Determine split options
    if args.mode:
        generate_single_split = args.mode in ['single', 'both']
        generate_k_fold = args.mode in ['kfold', 'both']
        K_FOLDS = args.folds
    else:
        generate_single_split, generate_k_fold, K_FOLDS = prompt_split_options()
    
    if not os.path.exists(INPUT_FILE):
        print(f"\n[ERROR] Input file {INPUT_FILE} not found.")
        print("        Please ensure you are running from the project root.")
        return

    print("→ Loading dataset records...")

    # Step 1: Load master labels
    records = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 15:
                continue  # skip malformed lines
            image_name = parts[0]
            labels     = [int(x) for x in parts[1:]]
            records.append({'image_name': image_name, 'labels': labels})

    df = pd.DataFrame(records)
    if df.empty:
        print("ERROR: No valid records found in input file.")
        return

    # Extract patient ID and disease flags
    df['patient_id'] = df['image_name'].apply(lambda x: x.split('_')[0])
    df['has_pneumonia'] = df['labels'].apply(lambda l: l[PNEUMONIA_IDX] == 1)
    df['has_any_disease'] = df['labels'].apply(lambda l: sum(l) > 0)
    df['is_no_finding']   = df['labels'].apply(lambda l: sum(l) == 0)

    # Step 2: Build Pneumonia Pool (Prioritize "Pneumonia-only" images)
    pneumonia_only  = df[df['has_pneumonia'] & (df['labels'].apply(lambda l: sum(l) == 1))].copy()
    pneumonia_mixed = df[df['has_pneumonia'] & (df['labels'].apply(lambda l: sum(l) > 1))].copy()
    
    if (len(pneumonia_only) + len(pneumonia_mixed)) < TARGET_PNEUMONIA:
        pneumonia_df = pd.concat([pneumonia_only, pneumonia_mixed], ignore_index=True)
    elif len(pneumonia_only) >= TARGET_PNEUMONIA:
        pneumonia_df = pneumonia_only.sample(n=TARGET_PNEUMONIA, random_state=RANDOM_SEED)
    else:
        needed = TARGET_PNEUMONIA - len(pneumonia_only)
        top_up = pneumonia_mixed.sample(n=needed, random_state=RANDOM_SEED)
        pneumonia_df = pd.concat([pneumonia_only, top_up], ignore_index=True)

    pneumonia_df['binary_label'] = 1

    # Step 3: Build Normal/Other Pool (Prioritize "No Finding" images)
    non_pneumonia_df = df[~df['has_pneumonia']].copy()
    no_finding = non_pneumonia_df[non_pneumonia_df['is_no_finding']].copy()
    other_disease = non_pneumonia_df[non_pneumonia_df['has_any_disease']].copy()

    if len(no_finding) >= TARGET_NORMAL:
        normal_df = no_finding.sample(n=TARGET_NORMAL, random_state=RANDOM_SEED)
    else:
        needed = TARGET_NORMAL - len(no_finding)
        top_up = other_disease.sample(n=min(needed, len(other_disease)), random_state=RANDOM_SEED)
        normal_df = pd.concat([no_finding, top_up], ignore_index=True)

    normal_df['binary_label'] = 0
    
    # Combine into final dataset
    full_df = pd.concat([pneumonia_df, normal_df], ignore_index=True)
    print(f"→ Composition complete: {len(pneumonia_df)} Positive, {len(normal_df)} Negative.")

    # Step 4: Generate K-Fold Splits (Patient-level stratification)
    if generate_k_fold:
        print(f"→ Generating {K_FOLDS}-fold stratified splits...")
        fold_dir = 'data/folds'
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
        os.makedirs(fold_dir, exist_ok=True)
        
        sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(full_df, full_df['binary_label'], groups=full_df['patient_id'])):
            train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
            test_fold_df  = full_df.iloc[test_idx].reset_index(drop=True)
            
            write_list(train_fold_df, f'{fold_dir}/train_fold{fold}.txt')
            write_list(test_fold_df,  f'{fold_dir}/test_fold{fold}.txt')
            
            p = test_fold_df['binary_label'].sum()
            n = len(test_fold_df) - p
            print(f"  Fold {fold}: [Test size: {len(test_fold_df):>3} (P:{p}, N:{n})] [Train size: {len(train_fold_df):>4}]")

        print(f"→ [SUCCESS] K-fold files saved to '{fold_dir}/'")

    # Step 5: Generate Single-Split (70/15/15)
    if generate_single_split:
        print("→ Generating single-split (70/15/15) files...")
        
        # Train (70%) vs Temp (30%)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
        train_idx, temp_idx = next(gss1.split(full_df, groups=full_df['patient_id']))

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        temp_df  = full_df.iloc[temp_idx].reset_index(drop=True)

        # Val (15%) vs Test (15%) from Temp
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['patient_id']))

        val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)

        write_list(train_df, 'data/train_list.txt')
        write_list(val_df,   'data/val_list.txt')
        write_list(test_df,  'data/test_list.txt')

        print("\nDATASET SPLIT SUMMARY:")
        print_stats_table({"Train": train_df, "Val": val_df, "Test": test_df})
        print("→ [SUCCESS] Single-split files saved to 'data/'")

if __name__ == "__main__":
    main()
