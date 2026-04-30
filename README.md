# DINOv3 Remote Sensing Scene Classification - CSCU9Z7 Dissertation

Author: Zhou Zirui  
Stirling ID: 3147943  
Supervisor: Dr. Zaid Al-Huda

## 1. Project Description

This project investigates remote sensing scene understanding on the MLRSNet dataset using a local DINOv3 ViT-S/16 backbone. The code trains and evaluates a primary scene classifier, several secondary multilabel classifiers, and a cascade pipeline where the predicted primary scene is used as the condition for secondary semantic label prediction.

## 2. Folder Structure

- `src/` - project source code for data loading, data splitting, model definitions, losses, training scripts, and evaluation scripts.
- `dinov3/` - third-party DINOv3 source code used as the local backbone implementation.
- `train_data/` - dataset instructions and expected local dataset location.
- `model/` - model weight directory and checkpoint path notes.
- `results/` - small result tables and figures produced by the evaluation scripts.
- `requirements.txt` - pinned Python dependencies used by the project environment.

The submitted code folder has the following structure:

```text
.
|-- README.md
|-- requirements.txt
|-- src/
|-- dinov3/
|-- train_data/
|   `-- README.md
|-- model/
|   `-- README.md
`-- results/
```

## 3. Code Provenance

The following files are project code written for this dissertation:

- `src/dataset_loader.py`
- `src/dataset_splitter.py`
- `src/dinov3_backbone.py`
- `src/dinov3_primary_classifier.py`
- `src/dinov3_conditional_multilabel.py`
- `src/dinov3_secondary_baselines.py`
- `src/multilabel_losses.py`
- `src/train_*.py`
- `src/evaluate_*.py`

Third-party code:

- `dinov3/` is the DINOv3 repository from Meta. It is included because the project loads the backbone through `torch.hub.load(..., source="local")`. See the original DINOv3 project and license in `dinov3/README.md` and `dinov3/LICENSE.md`.
- The MLRSNet dataset is not included in this ZIP. Dataset citation:
  Qi, Xiaoman, et al. "MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding." ISPRS Journal of Photogrammetry and Remote Sensing 169 (2020): 337-350.
- Python libraries used by the project are listed in `requirements.txt`.

## 4. Dataset

Download MLRSNet from Mendeley Data:

https://data.mendeley.com/datasets/7j9bv9vwsx/3

After downloading and extracting the dataset, place it in the project root using this structure:

```text
train_data/
|-- Images/
|   |-- airport/
|   |-- airplane/
|   `-- ...
`-- labels/
    |-- airport.csv
    |-- airplane.csv
    `-- ...
```

The expected dataset contains 46 primary scene folders and 46 CSV label files. The original dataset has 109,161 remote sensing images and 60 predefined secondary semantic labels.

Generate the JSON data splits with:

```powershell
.\venv\Scripts\python.exe src/dataset_splitter.py
```

The default split ratio is 40% training, 10% validation, and 50% testing. The generated files are:

- `data_splits/train.json`
- `data_splits/val.json`
- `data_splits/test.json`

## 5. Model Weights

The code requires the DINOv3 ViT-S/16 pretrained checkpoint. Place it locally at:

```text
model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

The trained dissertation checkpoints are expected in `model/` or `ablation_models/` when running evaluation scripts. Recreate them by running the training scripts, or place previously trained checkpoints at the paths shown in each evaluation script.

Main expected checkpoint paths:

- `model/best_model_primary.pth`
- `model/best_model_conditional_multilabel_aslgb.pth`
- `model/best_model_conditional_multilabel_asl.pth`
- `ablation_models/dinov3_bce/best_model_multilabel_bce.pth`
- `ablation_models/dinov3_bce_correlation/best_model_multilabel_bce_correlation.pth`
- `ablation_models/dinov3_asl_independent/best_model_multilabel_asl_independent.pth`
- `ablation_models/dinov3_aslgb_independent/best_model_multilabel_aslgb_independent.pth`

## 6. Build and Run

Create and activate a virtual environment:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124` wheels are not available from the default index, install them from the PyTorch CUDA 12.4 wheel index:

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the pipeline from the project root.

Make the `src/` folder importable for this PowerShell session:

```powershell
$env:PYTHONPATH = "src"
```

Generate dataset splits:

```powershell
.\venv\Scripts\python.exe src/dataset_splitter.py
```

Train the primary classifier:

```powershell
.\venv\Scripts\python.exe src/train_primary_classifier.py
```

Train the main conditional secondary multilabel model:

```powershell
.\venv\Scripts\python.exe src/train_conditional_multilabel_aslgb.py
```

Train ablation models:

```powershell
.\venv\Scripts\python.exe src/train_secondary_bce_baseline.py
.\venv\Scripts\python.exe src/train_secondary_asl_baseline.py
.\venv\Scripts\python.exe src/train_secondary_aslgb_baseline.py
.\venv\Scripts\python.exe src/train_conditional_bce_correlation.py
```

Evaluate the primary classifier:

```powershell
.\venv\Scripts\python.exe src/evaluate_primary_classifier.py
```

Evaluate secondary multilabel models:

```powershell
.\venv\Scripts\python.exe src/evaluate_conditional_multilabel_aslgb.py
.\venv\Scripts\python.exe src/evaluate_conditional_multilabel_asl.py
.\venv\Scripts\python.exe src/evaluate_conditional_bce_correlation.py
.\venv\Scripts\python.exe src/evaluate_secondary_bce_baseline.py
.\venv\Scripts\python.exe src/evaluate_secondary_asl_baseline.py
.\venv\Scripts\python.exe src/evaluate_secondary_aslgb_baseline.py
```

Evaluate the cascade pipeline:

```powershell
.\venv\Scripts\python.exe src/evaluate_primary_secondary_cascade.py
```

## 7. Outputs

Training scripts save checkpoints to `model/` or `ablation_models/`. Evaluation scripts write CSV reports to `results/eval_report_*` folders in the working project. The submission package includes only small example result files under `results/`.

Common output files:

- `summary_metrics.csv` - overall evaluation metrics.
- `per_class_metrics.csv` - per-label metrics.
- `combined_per_class_metrics.csv` - cascade metrics in the combined primary and secondary label space.

## 8. Configuration Notes

The scripts do not use command-line argument parsing. Default paths, batch sizes, learning rates, thresholds, and epoch counts are set in the `if __name__ == "__main__":` block at the bottom of each script.

The default DataLoader setting is `num_workers=4`. If the machine has limited memory or Windows multiprocessing fails, change `num_workers` to `0` in the relevant script.
