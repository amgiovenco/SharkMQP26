import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*80)
print("DEVICE INFORMATION")
print("="*80)
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: CUDA not available! Running on CPU (will be very slow)")
print("="*80)
print()

# --- Dataset Class --- 
class SharkFinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- TCN Architecture --- 
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

# --- Training function --- 
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_loader), f1, acc

# --- Evaluation function ---
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(data_loader), f1, acc, all_preds, all_labels

# --- Load best hyperparameters --- 
print("LOADING BEST HYPERPARAMETERS")
print("="*80)
with open('best_hyperparameters_real.json', 'r') as f:
    best_config = json.load(f)

print(f"Trial Number: {best_config['trial_number']}")
print(f"Validation F1 Score: {best_config['f1_score']:.4f}")
print("\nHyperparameters:")
for key, value in best_config['hyperparameters'].items():
    print(f"  {key}: {value}")
print("="*80 + "\n")

# --- Load and prepare data --- 
print("LOADING REAL DATA")
print("="*80)

df = pd.read_csv('shark_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Species'])
X = df.iloc[:, 1:].values
num_classes = len(label_encoder.classes_)

print(f"Number of classes: {num_classes}")
print(f"Classes: {list(label_encoder.classes_)}\n")

# Split data BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Normalize AFTER split (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape both
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

print(f"\nFinal shapes:")
print(f"  Training: {X_train.shape}")
print(f"  Test: {X_test.shape}")
print("="*80 + "\n")

# Create datasets
params = best_config['hyperparameters']
train_dataset = SharkFinDataset(X_train, y_train)
test_dataset = SharkFinDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

# Build model with best hyperparameters
print("BUILDING MODEL")
print("="*80)

# Generate channel list
num_layers = params['num_layers']
base_filters = params['base_filters']
filter_growth = params['filter_growth']

if filter_growth == 'constant':
    num_channels = [base_filters] * num_layers
elif filter_growth == 'linear':
    num_channels = [base_filters * (i + 1) for i in range(num_layers)]
else:  # exponential
    num_channels = [base_filters * (2 ** i) for i in range(num_layers)]
    num_channels = [min(ch, 512) for ch in num_channels]

print(f"Architecture: {num_channels}")

model = TemporalConvNet(
    num_inputs=1,
    num_channels=num_channels,
    num_classes=num_classes,
    kernel_size=params['kernel_size'],
    dropout=params['dropout'],
    reverse_dilation=params['reverse_dilation']
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print("="*80 + "\n")

# Setup training
criterion = nn.CrossEntropyLoss()

if params['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                          weight_decay=params['weight_decay'])
elif params['optimizer'] == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], 
                           weight_decay=params['weight_decay'])
else:  # SGD
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], 
                         momentum=0.9, weight_decay=params['weight_decay'])

# --- Train model --- 
print("TRAINING FINAL MODEL (REAL DATA ONLY)")
print("="*80)

max_epochs = 200
patience = 30
best_train_f1 = 0
epochs_no_improve = 0

for epoch in range(max_epochs):
    train_loss, train_f1, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    if train_f1 > best_train_f1:
        best_train_f1 = train_f1
        epochs_no_improve = 0
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': params,
            'architecture': {
                'num_channels': num_channels,
                'total_parameters': total_params
            },
            'epoch': epoch + 1,
            'train_f1': train_f1,
            'train_acc': train_acc
        }, 'best_tcn_model_real_only.pth')
    else:
        epochs_no_improve += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Acc: {train_acc:.4f}")
    
    if epochs_no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest training F1: {best_train_f1:.4f}")
print("="*80 + "\n")

# Load best model and evaluate on test set
checkpoint = torch.load('best_tcn_model_real_only.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print("EVALUATING ON TEST SET (REAL DATA ONLY)")
print("="*80)
test_loss, test_f1, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1 Score: {test_f1:.4f}")
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(test_labels, test_preds, 
                          target_names=label_encoder.classes_,
                          digits=4))

print("="*80)
print("CONFUSION MATRIX")
print("="*80)
cm = confusion_matrix(test_labels, test_preds)
print(cm)

# Save results
results = {
    'test_f1': float(test_f1),
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'best_train_f1': float(best_train_f1),
    'validation_f1': best_config['f1_score'],
    'data_info': {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_classes': num_classes,
        'data_type': 'real_only'
    },
    'hyperparameters': params,
    'model_architecture': {
        'num_channels': num_channels,
        'total_parameters': total_params
    }
}

with open('final_test_results_real_only.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print("Model saved to: best_tcn_model_real_only.pth")
print("Results saved to: final_test_results_real_only.json")
print("\nDATA USED:")
print("  - Training: Real data only")
print("  - Testing: Real data only")
print("  - NO synthetic data used")
print("="*80)