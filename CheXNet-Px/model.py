"""
CheXNet Model Architecture (Binary Classification)
--------------------------------------------------
This script defines the neural network architecture for binary pneumonia detection.
It adapts the 14-class DenseNet121 (CheXNet) by remapping its weights and 
replacing its final layer with a binary classification head.

Key Logic:
- Loads pretrained CheXNet weights from 'model.pth.tar'.
- Remaps state-dict keys to resolve naming conflicts between PyTorch versions.
- Replaces the 14-label head with a single-output linear layer for binary classification.
"""

import os
import re
import torch
import torch.nn as nn
import torchvision

from read_data import get_device

# --- Model Definition ---

class DenseNet121Binary(nn.Module):
    """
    Binary Pneumonia Classifier based on DenseNet121.
    
    The model starts with the arnoweng CheXNet weights (14 classes) and is 
    finetuned to distinguish between 'Pneumonia' and 'Normal'.
    """
    def __init__(self):
        super(DenseNet121Binary, self).__init__()

        # 1. Initialize standard DenseNet121 architecture
        self.densenet121 = torchvision.models.densenet121(weights=None)
        num_ftrs = self.densenet121.classifier.in_features
        
        # Temporary 14-class classifier to match the structure of the pretrained weights
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )

        # 2. Load and Remap Pretrained Weights
        ckpt_path = 'model.pth.tar'
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Critical Error: CheXNet backbone weights not found at {ckpt_path}")
        
        # Load weights to CPU to avoid CUDA/MPS memory issues during initialization
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']

        # Weight Remapping Logic:
        # - Removes 'module.' prefix (from DataParallel).
        # - Fixes 'norm.1' -> 'norm1' and 'conv.1' -> 'conv1' naming differences 
        #   between modern torchvision and the original CheXNet checkpoint.
        new_state_dict = {
            re.sub(r'(norm|conv)\.([12])', r'\1\2', k.replace('module.', '', 1)): v 
            for k, v in state_dict.items()
        }
        
        self.load_state_dict(new_state_dict)
        print("→ Pretrained CheXNet weights loaded successfully.")

        # 3. Final Adaptation: Binary Head
        # Replace the 14-class head with a single output (raw logits).
        # Backbone remains unfrozen for end-to-end finetuning.
        self.densenet121.classifier = nn.Linear(num_ftrs, 1)

        # 4. Print Architecture Summary
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print("\n" + "-"*40)
        print(f"{'MODEL ARCHITECTURE SUMMARY':^40}")
        print("-"*40)
        print(f" Total Parameters:     {total_params:,}")
        print(f" Trainable Parameters: {trainable_params:,}")
        print(f" Training Mode:        Full Network")
        print(f" Trainable %:          {(trainable_params / total_params) * 100:.2f}%")
        print("-"*40 + "\n")

    def forward(self, x):
        """Standard forward pass through the backbone and binary head."""
        return self.densenet121(x)

# --- Loader Utility ---

def load_model():
    """
    Utility function to instantiate the model and move it to the detected hardware accelerator.
    
    Returns:
        tuple: (model, device)
    """
    device = get_device()
    print(f"→ Initializing model on {device}...")
    
    model = DenseNet121Binary()
    model = model.to(device)
    
    return model, device

if __name__ == "__main__":
    # Test model initialization
    load_model()