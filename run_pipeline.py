"""
CheXNet Pipeline Orchestrator
-----------------------------
This script serves as the main entry point for the CheXNet binary classification pipeline.
It coordinates dataset preparation, model training, post-hoc calibration, and evaluation.

Usage:
    python run_pipeline.py [--mode {single, kfold}] [--folds N]
"""

import subprocess
import sys
import os
import argparse

# --- Utility Functions ---

def run_command(step_num, step_name, cmd_list):
    """
    Executes a shell command as part of a pipeline step.
    
    Args:
        step_num (int): The current step number in the pipeline.
        step_name (str): A descriptive name for the current step.
        cmd_list (list): The command and its arguments to be executed via subprocess.
    """
    print("\n" + "="*60)
    print(f" [{step_num}/4] {step_name}")
    print("="*60)
    print(f"→ Executing: {' '.join(cmd_list)}\n")
    
    try:
        # Run the command and wait for completion.
        # check=True ensures that a CalledProcessError is raised if the command fails.
        subprocess.run(cmd_list, check=True)
        print(f"\n→ {step_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step_name} failed with exit code {e.returncode}")
        print("Pipeline aborted.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline stopped by user.")
        sys.exit(1)

# --- Main Orchestration Logic ---

def main():
    """
    Main entry point for the pipeline. Handles configuration, argument parsing, 
    and step execution for both 'single-split' and 'k-fold' modes.
    """
    print("\n" + "#"*60)
    print(f"{'CHEXNET BINARY CLASSIFICATION PIPELINE (ORCHESTRATOR)':^60}")
    print("#"*60)

    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Automated CheXNet Pipeline Orchestrator")
    parser.add_argument('--mode', type=str, choices=['single', 'kfold'], 
                        help="Pipeline mode: 'single' (70/15/15 split) or 'kfold' (cross-validation)")
    parser.add_argument('--folds', type=int, help="Number of folds for k-fold mode (default: 5)")
    parser.add_argument('--skip-training', action='store_true', help="Skip the model training step")
    args = parser.parse_args()

    # Interactive Training Decision (if not specified via CLI)
    skip_training = args.skip_training
    if not any(arg in sys.argv for arg in ['--skip-training']):
        print("\n→ TRAINING SELECTION")
        choice = input("Do you want to train the model? (y/n) [default: y]: ").strip().lower() or 'y'
        skip_training = (choice == 'n')

    # Interactive Configuration (if CLI arguments are missing)
    if not args.mode:
        print("\n→ PIPELINE CONFIGURATION")
        print("1. Single-split mode (Train/Val/Test)")
        print("2. K-fold cross-validation mode")
        while True:
            choice = input("\nSelect mode (1 or 2): ").strip()
            if choice == '1':
                args.mode = 'single'
                args.folds = 1
                break
            elif choice == '2':
                args.mode = 'kfold'
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    # Fold selection for k-fold mode
    if args.mode == 'kfold' and args.folds is None:
        while True:
            try:
                args.folds = int(input("Enter number of folds (default 5): ").strip() or '5')
                if args.folds < 2:
                    print("Error: K-folds must be at least 2.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    print(f"\n→ Pipeline initialized in {args.mode.upper()} mode.")
    if skip_training:
        print("→ [INFO] Training step will be skipped.")

    # --- Step 1: Dataset Preparation ---
    # Generates .txt files containing image names and binary labels.
    run_command(1, "DATASET PREPARATION", [
        sys.executable, 'build_splits.py', 
        '--mode', args.mode, 
        '--folds', str(args.folds)
    ])

    # --- Step 2: Model Training ---
    # Fine-tunes DenseNet121 for binary classification (Pneumonia vs. Normal).
    if not skip_training:
        run_command(2, "MODEL TRAINING", [
            sys.executable, 'train_model.py', 
            '--mode', args.mode, 
            '--folds', str(args.folds)
        ])
    else:
        print("\n" + "="*60)
        print(" [2/4] MODEL TRAINING - SKIPPED")
        print("="*60)

    # --- Step 3 & 4: Calibration and Evaluation ---
    if args.mode == 'single':
        # Single-split calibration and evaluation
        run_command(3, "POST-HOC CALIBRATION", [sys.executable, 'calibrate.py', '--mode', 'single'])
        run_command(4, "PERFORMANCE EVALUATION", [sys.executable, 'evaluate.py', '--mode', 'single'])
    else:
        # Batch processing for k-fold cross-validation
        print(f"\n" + "="*60)
        print(f" [3-4/4] BATCH CALIBRATION & EVALUATION ({args.folds} folds)")
        print("="*60)
        for i in range(args.folds):
            print(f"\n→ PROCESSING FOLD {i}...")
            # Calibrate each fold to optimize probability estimates
            subprocess.run([sys.executable, 'calibrate.py', '--mode', 'kfold', 
                            '--folds', str(args.folds), '--fold-idx', str(i)], check=True)
            # Evaluate each fold and compare against baseline metrics
            subprocess.run([sys.executable, 'evaluate.py', '--mode', 'kfold', 
                            '--folds', str(args.folds), '--fold-idx', str(i)], check=True)

    print("\n" + "#"*60)
    print(f"{'PIPELINE COMPLETE - ALL ARCHITECTURAL STEPS FINISHED':^60}")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
