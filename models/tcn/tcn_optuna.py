import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings('ignore')

# Set device - DirectML for AMD GPU support on Windows
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using device: DirectML (AMD GPU)")
    print(f"DirectML device name: {torch_directml.device_name(0)}")
except ImportError:
    print("torch-directml not found. Install with: pip install torch-directml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Falling back to: {device}")
except Exception as e:
    print(f"DirectML initialization failed: {e}")
    device = torch.device('cpu')
    print(f"Falling back to CPU")

# ============================================================================
# DATASET CLASS WITH AUGMENTATION
# ============================================================================
class SharkFinDataset(Dataset):
    """Custom Dataset for shark fin fluorescence time-series data with augmentation"""
    def __init__(self, features, labels, augment=False):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.augment:
            # Small random noise
            noise = 0.01 * torch.randn_like(x)
            # Random scaling
            scale = 1 + 0.05 * torch.randn(1)
            x = x * scale + noise

            # Random horizontal shifts
            max_shift = 3
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
        
        return x, y

# ============================================================================
# TCN MODEL ARCHITECTURE
# ============================================================================
class CausalConv1d(nn.Module):
    """Causal 1D convolution with weight normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=self.padding, dilation=dilation)
        )
    
    def forward(self, x):
        x = self.conv(x)
        # Remove future information (causal)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class TemporalBlock(nn.Module):
    """Temporal block with residual connections and batch normalization"""
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
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for time-series classification"""
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=3, dropout=0.2, reverse_dilation=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            if reverse_dilation:
                # Reverse dilation: 16 -> 8 -> 4 -> 2 -> 1 (for 5 layers)
                dilation_size = 2 ** (num_levels - i - 1)
            else:
                # Standard dilation: 1 -> 2 -> 4 -> 8 -> 16
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
        # x shape: (batch, features, seq_len)
        y = self.network(x)
        # Global average pooling
        y = torch.mean(y, dim=2)
        return self.fc(y)

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
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
    return total_loss / len(train_loader), f1

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on dataset"""
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
    return total_loss / len(data_loader), f1

# ============================================================================
# LOAD AND PREPARE DATA (GLOBAL - DONE ONCE)
# ============================================================================
print("=" * 80)
print("LOADING AND PREPROCESSING DATA")
print("=" * 80)

# Load data
df = pd.read_csv('shark_data.csv')  # Replace with your actual filename
print(f"Dataset shape: {df.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Species'])
X = df.iloc[:, 1:].values
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Fixed test split with seed 8
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=8, stratify=y
)

print(f"Training pool size: {len(X_train_full)}")
print(f"Test set size: {len(X_test)}\n")

# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================
def objective(trial):
    """
    Optuna objective function to optimize hyperparameters
    Returns: Macro F1 score on validation set
    """
    
    # ========================
    # HYPERPARAMETER SEARCH SPACE
    # ========================
    
    # Architecture parameters
    num_layers = trial.suggest_int('num_layers', 3, 7)
    base_filters = trial.suggest_categorical('base_filters', [32, 64, 128])
    filter_growth = trial.suggest_categorical('filter_growth', ['constant', 'linear', 'exponential'])
    
    # Generate channel list based on growth pattern
    if filter_growth == 'constant':
        num_channels = [base_filters] * num_layers
    elif filter_growth == 'linear':
        num_channels = [base_filters * (i + 1) for i in range(num_layers)]
    else:  # exponential
        num_channels = [base_filters * (2 ** i) for i in range(num_layers)]
        # Cap maximum channels
        num_channels = [min(ch, 512) for ch in num_channels]
    
    kernel_size = trial.suggest_int('kernel_size', 3, 15, step=2)  # Odd numbers only
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    reverse_dilation = trial.suggest_categorical('reverse_dilation', [True, False])
    
    # Training parameters
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    
    # Data augmentation
    use_augmentation = trial.suggest_categorical('use_augmentation', [True, False])
    
    # Learning rate scheduler
    use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
    if use_scheduler:
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 15)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.7)
    
    # ========================
    # PREPARE DATA SPLIT
    # ========================
    
    # Use a fixed validation seed for fair comparison across trials
    val_seed = 42
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=val_seed, stratify=y_train_full
    )
    
    # Create datasets
    train_dataset = SharkFinDataset(X_train, y_train, augment=use_augmentation)
    val_dataset = SharkFinDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ========================
    # BUILD MODEL
    # ========================
    
    torch.manual_seed(val_seed)
    model = TemporalConvNet(
        num_inputs=1,
        num_channels=num_channels,
        num_classes=num_classes,
        kernel_size=kernel_size,
        dropout=dropout,
        reverse_dilation=reverse_dilation
    ).to(device)
    
    # ========================
    # SETUP TRAINING
    # ========================
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience
        )
    
    # ========================
    # TRAINING LOOP
    # ========================
    
    max_epochs = 100  # Reduced for faster hyperparameter search
    patience = 20
    best_val_f1 = 0
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        if use_scheduler:
            scheduler.step(val_f1)
        
        # Track best validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Report intermediate value for pruning
        trial.report(val_f1, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if epochs_no_improve >= patience:
            break
    
    return best_val_f1

# ============================================================================
# RUN OPTUNA OPTIMIZATION
# ============================================================================
print("=" * 80)
print("STARTING OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 80)
print(f"Target: Maximize Macro F1 Score")
print(f"Number of trials: 200")
print(f"Database: optuna_tcn_study.db")
print("=" * 80)

# Create or load study
study_name = "tcn_shark_classification"
storage_name = "sqlite:///optuna_tcn_study.db"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',  # Maximize F1 score
    load_if_exists=True,   # Resume if study exists
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
        interval_steps=1
    )
)

print(f"\nStudy created/loaded: {study_name}")
print(f"Storage: {storage_name}")
print(f"Optimization direction: maximize\n")

# Run optimization
study.optimize(
    objective,
    n_trials=200,
    timeout=None,
    n_jobs=1,  # Set to >1 for parallel trials if you have multiple GPUs
    show_progress_bar=True,
    callbacks=[
        lambda study, trial: print(
            f"\nTrial {trial.number} finished with F1 Score: {trial.value:.4f}"
        )
    ]
)

# ============================================================================
# PRINT RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)

print(f"\nNumber of finished trials: {len(study.trials)}")

# Best trial
print("\n" + "-" * 80)
print("BEST TRIAL")
print("-" * 80)
trial = study.best_trial
print(f"  Trial number: {trial.number}")
print(f"  Macro F1 Score: {trial.value:.4f}")
print("\n  Best Hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Pruned trials
pruned_trials = study.get_trials(states=[TrialState.PRUNED])
complete_trials = study.get_trials(states=[TrialState.COMPLETE])
print(f"\n  Statistics:")
print(f"    Complete trials: {len(complete_trials)}")
print(f"    Pruned trials: {len(pruned_trials)}")

# Top 10 trials
print("\n" + "-" * 80)
print("TOP 10 TRIALS")
print("-" * 80)
top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)[:10]
for i, t in enumerate(top_trials, 1):
    if t.value is not None:
        print(f"{i}. Trial {t.number}: F1 = {t.value:.4f}")

# Parameter importance (requires optuna-integration if available)
try:
    import optuna.importance
    print("\n" + "-" * 80)
    print("HYPERPARAMETER IMPORTANCE")
    print("-" * 80)
    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"  {param}: {imp:.4f}")
except:
    print("\n(Install optuna-integration for parameter importance analysis)")

print("\n" + "=" * 80)
print("RESULTS SAVED TO DATABASE")
print("=" * 80)
print(f"Database location: optuna_tcn_study.db")
print(f"Study name: {study_name}")
print("\nTo visualize results, use Optuna Dashboard:")
print("  pip install optuna-dashboard")
print(f"  optuna-dashboard {storage_name}")
print("=" * 80)

# ============================================================================
# SAVE BEST HYPERPARAMETERS TO FILE
# ============================================================================
import json

best_params_file = 'best_hyperparameters.json'
with open(best_params_file, 'w') as f:
    json.dump({
        'trial_number': trial.number,
        'f1_score': trial.value,
        'hyperparameters': trial.params
    }, f, indent=4)

print(f"\nBest hyperparameters also saved to: {best_params_file}")