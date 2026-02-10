"""
Create TCN inference bundle from trained model.

This script packages the trained TCN model, scaler, and label encoder
into a single pickle file for production inference.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
 

# --- TCN Architecture (must match training) ---
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=self.padding, dilation=dilation)
        )
    
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=3, dropout=0.2, reverse_dilation=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            if reverse_dilation:
                dilation_size = 2 ** (num_levels - i - 1)
            else:
                dilation_size = 2 ** i
            
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, 
                            dilation_size, dropout)
            )
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        y = self.network(x)
        y = torch.mean(y, dim=2)
        return self.fc(y)


def create_tcn_bundle(
    model_checkpoint_path: str = 'best_tcn_model_real_only.pth',
    training_data_path: str = 'shark_dataset.csv',
    output_path: str = 'model/tcn_bundle.pkl'
):
    """
    Create inference bundle from trained TCN model.

    Args:
        model_checkpoint_path: Path to trained model .pth file
        training_data_path: Path to training data CSV (for scaler and label encoder)
        output_path: Where to save the bundle
    """
    print("="*80)
    print("CREATING TCN INFERENCE BUNDLE")
    print("="*80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load checkpoint
    print(f"Loading model checkpoint from {model_checkpoint_path}...")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)

    # Extract configuration
    num_channels = checkpoint['architecture']['num_channels']
    hyperparameters = checkpoint['hyperparameters']
    
    print(f"Model architecture: {num_channels}")
    print(f"Hyperparameters: {hyperparameters}\n")

    # Load training data to fit scaler and label encoder
    print(f"Loading training data from {training_data_path}...")
    df = pd.read_csv(training_data_path)
    
    # Fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Species'])
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}\n")
    
    # Fit scaler on training features
    X = df.iloc[:, 1:].values
    scaler = StandardScaler()
    scaler.fit(X)
    
    print(f"Scaler fitted on {X.shape[0]} samples with {X.shape[1]} features\n")

    # Reconstruct model architecture
    print("Reconstructing model architecture...")
    model = TemporalConvNet(
        num_inputs=1,
        num_channels=num_channels,
        num_classes=num_classes,
        kernel_size=hyperparameters['kernel_size'],
        dropout=hyperparameters['dropout'],
        reverse_dilation=hyperparameters['reverse_dilation']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to CPU for portability
    model = model.cpu()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}\n")

    # Load results if available
    results_path = 'final_test_results_real_only.json'
    test_f1 = None
    test_accuracy = None
    
    if Path(results_path).exists():
        print(f"Loading test results from {results_path}...")
        with open(results_path, 'r') as f:
            results = json.load(f)
            test_f1 = results.get('test_f1')
            test_accuracy = results.get('test_accuracy')
            print(f"Test F1: {test_f1:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # Create bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'hyperparameters': hyperparameters,
        'architecture': {
            'num_channels': num_channels,
            'total_parameters': total_params
        },
        'test_f1': test_f1,
        'test_accuracy': test_accuracy,
        'num_classes': num_classes,
        'training_info': {
            'epoch': checkpoint.get('epoch'),
            'train_f1': checkpoint.get('train_f1'),
            'train_acc': checkpoint.get('train_acc')
        }
    }

    # Save bundle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving bundle to {output_path}...")
    joblib.dump(bundle, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Bundle saved successfully! ({file_size_mb:.2f} MB)")

    print("\n" + "="*80)
    print("BUNDLE CREATION COMPLETE")
    print("="*80)
    print(f"\nBundle contents:")
    print(f"  - Trained TCN model ({total_params:,} parameters)")
    print(f"  - StandardScaler (fitted on training data)")
    print(f"  - LabelEncoder ({num_classes} classes)")
    print(f"  - Hyperparameters and architecture info")
    print(f"  - Test performance metrics")
    print(f"\nTo use this bundle:")
    print(f"  1. Place it in your worker/model/ directory")
    print(f"  2. Update worker.py to import from tcn_inference:")
    print(f"     from SharkMQP26.backend.worker.tcn_inference import run_inference as ml_inference")
    print(f"  3. Run inference with the same interface as CNN")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create TCN inference bundle')
    parser.add_argument('--model', type=str, default='best_tcn_model_real_only.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='shark_dataset.csv',
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='model/tcn_bundle.pkl',
                       help='Output path for bundle')
    
    args = parser.parse_args()
    
    create_tcn_bundle(
        model_checkpoint_path=args.model,
        training_data_path=args.data,
        output_path=args.output
    )