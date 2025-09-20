# CNN-TASK

A small image classification project using a convolutional neural network (CNN). This repository contains code, dataset splits, and utilities used to train and evaluate a CNN model on a custom dataset.

---

## Table of contents

- Project overview
- Repository structure
- Requirements
- Setup (Windows PowerShell)
- Running the code
  - From `app.py` (script)
  - From the notebook `CODE.ipynb`
- Dataset format
- Training and evaluation notes
- Tips and troubleshooting
- License & acknowledgements

## Project overview

This repo trains a convolutional neural network (CNN) to classify images into multiple classes. It includes a training script (`app.py`), a Jupyter notebook with exploratory analysis and experiments (`CODE.ipynb`), CSV files describing training and testing splits, and example images under `DATASET/test` and `DATASET/train`.

The project uses TensorFlow/Keras for model implementation and training.

## Repository structure

- `app.py` - Main training/evaluation script (entrypoint). Review top of file for configurable CLI options.
- `CODE.ipynb` - Notebook with EDA, preprocessing, and experimentation.
- `requirements.txt` - Python dependencies used by the project.
- `class_counts.csv` - (Optional) counts of class examples.
- `le.npy` - Saved label encoder/lookup used by the code.
- `DATASET/Training_set.csv` - CSV that lists training images and labels.
- `DATASET/Testing_set.csv` - CSV that lists testing images and labels.
- `DATASET/train/` - Training images directory.
- `DATASET/test/` - Testing images directory.

## Requirements

This project was developed on Windows 10/11 with Python 3.8+.

Minimal dependencies (see `requirements.txt`):

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imblearn
- tensorflow
- Pillow
- opencv-python
- tqdm

It's recommended to create a virtual environment.

## Setup (Windows PowerShell)

1. Open PowerShell and change to the project directory:

```powershell
Set-Location -Path "d:\AI WORKSHOP\TASK\CNN-TASK"
```

2. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

If you get an execution policy error when activating, run PowerShell as Administrator and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip; python -m pip install -r requirements.txt
```

## Running the code

There are two main ways to run experiments: using `app.py` (script) or interactively from `CODE.ipynb`.

### Option A — Run training/eval script (`app.py`)

1. Review `app.py` for any configurable flags at the top (e.g., paths, hyperparameters).
2. Run the script from PowerShell:

```powershell
python .\app.py
```

Add flags if `app.py` supports command-line arguments. If you need help locating or modifying CLI options, open `app.py` and look for `argparse` usage or hard-coded paths.

### Option B — Use the notebook (`CODE.ipynb`)

1. Start Jupyter Lab / Notebook:

```powershell
python -m notebook
```

2. Open `CODE.ipynb` in the browser and run cells interactively. The notebook contains data loading, preprocessing, model definition, and training/evaluation cells.

## Dataset format

- `DATASET/Training_set.csv` and `DATASET/Testing_set.csv` should contain at least two columns: `image` (relative path or filename) and `label` (class name or numeric label).
- Images live under `DATASET/train/` and `DATASET/test/` respectively. Filenames should match those listed in CSVs.

If your CSVs use different column names, update the code in `app.py` and `CODE.ipynb` accordingly.

## Training and evaluation notes

- The code uses Keras/TensorFlow. If you have a GPU available, install the GPU build of TensorFlow for faster training.
- Class imbalance handling: the repository includes `imblearn` in `requirements.txt`; look in `CODE.ipynb` for oversampling or class-weight usage.
- Checkpoints: if model checkpointing is enabled, look for a `models/` or `checkpoints/` directory and confirm paths before training.
- Label encoder (`le.npy`) is provided — the code may load it for consistent label mapping when evaluating.

## Tips and troubleshooting

- If imports fail, ensure the virtual environment is active and dependencies are installed.
- If image loading fails, verify the paths in CSV files are correct and that images exist under `DATASET/*`.
- If GPU isn't used, confirm your TensorFlow installation supports CUDA and that CUDA/cuDNN versions match your GPU drivers.

## Quick example

A minimal example to load the dataset in Python:

```python
import pandas as pd
from pathlib import Path

root = Path(r"d:\AI WORKSHOP\TASK\CNN-TASK\DATASET")
train_df = pd.read_csv(root / 'Training_set.csv')
print(train_df.head())
```

## License & acknowledgements

This repo is provided as-is for educational purposes. Add an appropriate license if you plan to share or publicize the project.

---

If you'd like, I can also:
- Add a small `Makefile` or PowerShell script to automate setup and training.
- Inspect `app.py`/`CODE.ipynb` and add exact CLI instructions or fix paths.
- Add example trained model or scripts to export predictions.

If you want any of those, tell me which and I'll proceed.
