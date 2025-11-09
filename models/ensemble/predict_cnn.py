"""cnn/efficientnet model prediction."""

import glob
import numpy as np
import pandas as pd
from pathlib import Path
import io

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, RuntimeError):
    TORCH_AVAILABLE = False
    DEVICE = None

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    MATPLOTLIB_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    PIL_AVAILABLE = False


class CNNModel(nn.Module):
    """efficientnet-based cnn model."""
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        try:
            from torchvision import models
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        except (ImportError, RuntimeError) as e:
            raise ImportError("torchvision not available for cnn") from e

    def forward(self, x):
        return self.model(x)


def _generate_images(X_raw: pd.DataFrame):
    """generate pil images from fluorescence curves (matching training image generation)."""
    try:
        images = []
        temps = X_raw.columns.astype(float).values
        IMAGE_SIZE = (224, 224)
        DPI = 100

        for idx, row in X_raw.iterrows():
            # Match the exact image generation from cnn/scripts/generate_images.py
            fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI)

            # Plot with same color and linewidth as training
            ax.plot(temps, row.values, linewidth=2, color='#2E86AB')

            # Set limits to match training
            signal_values = row.values
            ax.set_xlim(temps.min(), temps.max())
            ax.set_ylim(signal_values.min() - 0.001, signal_values.max() + 0.001)

            # Remove axes and labels (matching training)
            ax.axis('off')

            # Remove all margins and padding (matching training)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                        facecolor='white', edgecolor='none', dpi=DPI)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            images.append(img)
            plt.close(fig)

        return images
    except Exception as e:
        print(f"failed to generate images: {e}")
        return None


def get_cnn_predictions(X_raw: pd.DataFrame, num_classes: int, models_dir: str = "./models") -> np.ndarray:
    """get cnn/efficientnet predictions."""
    print("  cnn/efficientnet...", end=" ", flush=True)
    if not TORCH_AVAILABLE or not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
        print("[FAIL] dependencies not available")
        return None

    try:
        files = glob.glob(f"{models_dir}/efficientnetb0_*.pth")
        if not files:
            print("[FAIL] not found")
            return None

        model_path = max(files, key=lambda p: Path(p).stat().st_mtime)
        model = CNNModel(num_classes=num_classes).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)

        # handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # if keys don't have 'model.' prefix, add it (training saves bare efficientnet)
        # but the CNNModel wrapper needs 'model.' prefix for its self.model
        fixed_state_dict = {}
        for key, val in state_dict.items():
            if not key.startswith('model.'):
                # add 'model.' prefix to match CNNModel.model
                new_key = 'model.' + key
                fixed_state_dict[new_key] = val
            else:
                fixed_state_dict[key] = val

        # load into the full CNNModel (which has the 'model.' structure)
        model.load_state_dict(fixed_state_dict, strict=False)

        model.eval()

        # generate images
        print("\n    generating images...", end=" ", flush=True)
        images = _generate_images(X_raw)
        if images is None:
            print("[FAIL]")
            return None

        print(f"[OK]")

        # transform and predict
        from torchvision import transforms as tv_transforms
        transform = tv_transforms.Compose([
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        images_tensor = torch.stack([transform(img) for img in images]).to(DEVICE)

        with torch.no_grad():
            outputs = model(images_tensor)
            proba = torch.softmax(outputs, dim=1)
            proba = proba.cpu().numpy()

        print(f"  cnn predictions: [OK] ({proba.shape})")
        return proba
    except Exception as e:
        print(f"[FAIL] error: {e}")
        return None