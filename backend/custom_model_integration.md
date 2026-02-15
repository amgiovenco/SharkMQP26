# Post-MQP Custom Model Integration
### Connor Jason

I have created a guide on how to integrate a custom model into our application so you can build on top of our work after our projet has concluded.

## Data Flow
```
User uploads CSV -> Backend API -> Redis Queue -> Worker Process -> Your Model -> Results
```

The worker process (`backend/worker/worker.py`) calls your model's `ml_inference()` function to make predictions on shark DNA melting curve data.

## Prerequisites

- Python 3.8+
- Comaptible trained model file(s)

### Dependencies
It is recommended to use a virtual environment to manage your dependencies. We currently use one in `backend/.venv`. If you have to create a new virtual environment, it is recommended to use the same file path to not cause issues in the service files.

If not already installed, install these dependencies:
```bash
pip install torch numpy pandas
```

If you need additional dependencies, install them:
```bash
pip install tensorflow
pip install scikit-learn
```

## Integration

### 1. Create Your Inference Script

Create a new file in the `backend/worker/` directory. For example: `worker/new_model_inference.py`

### 2. Implement `ml_inference()`

Your inference script must include a function with this exact function definition:

```python
def ml_inference(filepath: str, sample_index: int = 0, device: Optional[str] = None) -> dict:
```

Where the filepath is the path to the CSV file you wish to make an inference on, the sample_index is the row index to make a prediction on (these CSVs often contain multiple samples on separate rows), and device is either cuda or cpu to determine the compute hardware used.

Please refer to `backend/worker/inference_interface.py` for further instructions on how to implement this function.

### Verify Data Return Format

Your `ml_inference()` function must return a dictionary with this structure:

#### Success Response:
```python
{
    'success': True,
    'predictions': [
        {'rank': 1, 'species': 'Arabian smooth-hound', 'confidence': 0.95},
        {'rank': 2, 'species': 'Bull Shark', 'confidence': 0.03},
        {'rank': 3, 'species': 'Blacktip Shark', 'confidence': 0.02},
        # ... more predictions
    ],
    'sample_index': 0,
    'curve_data': {
        'frequencies': [60.0, 60.5, 61.0, ...], # Temperature values
        'signal': [0.123, 0.456, ...]  # Fluorescence values
    }
}
```

#### Error Response:
```python
{
    'success': False,
    'error': 'error message about what went wrong',
    'predictions': [],
    'sample_index': 0
}
```

### 3. Model Singleton

This isn't strictly required but it's recommended to implement the singleton pattern to reduce inference overhead. This will keep an instance of your model cached, eliminately the time to load and destory the model per inference. The pattern for my EfficientNet model looks something like this:
```python
# Global model instance
_model_instance = None

def get_model_instance() -> SharkSpeciesInference:
    """
    Get or create the singleton model instance.
    Loads model once and reuses it for all subsequent calls.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = SharkSpeciesInference()
    return _model_instance
```

Then, just call `get_model_instance()` where you would want to intiialize an instance of `SharkSpeciesInference`.

### 4. Update Worker Configuration

Edit `worker/worker.py` and change line 15 from:

```python
from worker.cnn_inference import ml_inference, CNNModel
```

To:

```python
from worker.new_model_inference import ml_inference, CustomModel
```

## Testing Your Model

It is recommended to test your implementation of this function locally before deploying it fully to production. I have included a test file for you to use in `backend/worker/test_inference.py`. Currently, it is setup to test my EfficientNet CNN model. All you have to do to use it for your model is change where it imports from. e.g.,

```python
from cnn_inference import ml_inference, CNNModel
```

turns into:

```python
from custom_model_inference import ml_inference, CustomModel
```

Even if it says importing your model object is an unused import, it's still required, at least for me.

Here is what it looks like when I run it, I would consider this a success and would then proceed to deployment:

```shell
(.venv) connorjason@vpnclient-10-217-64-245 worker % python test_inference.py 
Testing inference on: ../../data/shark_dataset.csv

Test 1/5 - Sample index 439
--------------------------------------------------------------------------------
Loading model from /Users/connorjason/VSCProjects/SharkMQP26/backend/worker/efficientnet/cnn_bundle.pkl...
Model loaded successfully!
  Device: cpu
  Classes: 57
  Species: ['Arabian smooth-hound', 'Atlantic Sharpnose shark', 'Blackchin guitarfish', 'Blacknose shark', 'Blackspotted smooth-hound', 'Blacktip reef shark', 'Blacktip shark', 'Blue shark', 'Bonnethead shark', 'Bowmouth guitarfish', 'Brownbanded bamboo shark', 'Bull shark', 'Caribbean reef shark', 'Common thresher shark', 'Copper shark', 'Dusky shark', 'Finetooth shark', 'Great hammerhead shark', 'Great white shark', 'Grey reef shark', 'Gulper shark', 'Gummy shark', 'Halavi guitarfish', 'Hooktooth shark', 'Japanese topeshark', 'Java shark', 'Lemon shark', 'Longtail stingray', 'Milk shark', 'Narrownose smooth-hound', 'Night shark', 'Nurse shark', 'Oceanic whitetip shark', 'Pacific bonnethead shark', 'Pacific guitarfish', 'Pacific smalltail shark', 'Pelagic thresher shark', 'Porbeagle shark', 'Roughskin dogfish', 'Sandbar shark', 'Sandtiger shark', 'Scalloped bonnethead shark', 'Scalloped hammerhead shark', 'Shortfin mako', 'Silky shark', 'Silvertip shark', 'Small tail shark', 'Smooth hammerhead shark', 'Spadenose stingray', 'Spinner shark', 'Spot-tail shark', 'Spotted Eagleray', 'Thornback ray', 'Tiger shark', 'Tope shark', 'Whitecheeck shark', 'Zebra shark']
  CV Accuracy: 97.75%
SUCCESS

Top 3:
  1. Sandbar shark                            0.9515
  2. Silvertip shark                          0.0054
  3. Spinner shark                            0.0051

Test 2/5 - Sample index 57
--------------------------------------------------------------------------------
SUCCESS

Top 3:
  1. Blacktip shark                           0.8947
  2. Night shark                              0.0101
  3. Silky shark                              0.0085

Test 3/5 - Sample index 2
--------------------------------------------------------------------------------
SUCCESS

Top 3:
  1. Arabian smooth-hound                     0.9643
  2. Copper shark                             0.0074
  3. Oceanic whitetip shark                   0.0052

Test 4/5 - Sample index 178
--------------------------------------------------------------------------------
SUCCESS

Top 3:
  1. Copper shark                             0.9433
  2. Bull shark                               0.0265
  3. Caribbean reef shark                     0.0035

Test 5/5 - Sample index 67
--------------------------------------------------------------------------------
SUCCESS

Top 3:
  1. Blacktip shark                           0.8147
  2. Brownbanded bamboo shark                 0.0236
  3. Longtail stingray                        0.0200
```

If it doesn't work, please verify your dependencies are installed, the file paths to model file(s) and CSV data are correct, etc.

## Deployments

SSH into the VM:

```shell
ssh kmlee@wildsense.wpi.edu
```

and locate the project. At the time of writing this, it's in `/srv/SharkMQP26` but it will likely be moved to `/opt/SharkMQP26` once I am "done" with it.

### 1.  Model Dependencies

Update `requirements.txt` if you have additional dependencies then reinstall:

```bash
source backend/.venv/bin/activate # change path if needed
pip install -r requirements.txt
```

### 2. Deploy Model Files

Place your model weights in `backend/worker/models/`:

```text
backend/
  worker/
    models/
      custom_model.pth # Your model weights
      config.json # Any optional config files
      config.pkl # Any optional config files
      config.yml # Any optional config files
    custom_model_inference.py # Your inference code
```

### 3. Restart the Worker Service

```bash
sudo systemctl restart sharkid-worker
```

Done!

You can now go to [wildsense.wpi.edu](https://wildsense.wpi.edu/), login, and test your inference out!

If you run into any issues, you can check the worker status with:

```bash
sudo systemctl status sharkid-worker
```

or its logs with:

```bash
sudo journalctl -u sharkid-worker -f
```