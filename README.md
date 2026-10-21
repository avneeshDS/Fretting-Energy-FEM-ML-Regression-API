# Finite-Element and Machine-Learning Framework for Fretting Contacts

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.rineng.2026.1113296-blue)](https://doi.org/10.1016/j.rineng.2026.1113296)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Abaqus](https://img.shields.io/badge/FEA-Abaqus-005386)](https://www.3ds.com/products/simulia/abaqus)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9)](https://docs.astral.sh/uv/)

Code and API implementation associated with the research article:

**Finite-element and machine-learning framework for dissipated-energy prediction and slip-regime classification in fretting contacts**

Avneesh Kumar, Marek Vostrak, and Sarka Houdkova

*Results in Engineering*, Volume 32, 2026, Article 1113296

[https://doi.org/10.1016/j.rineng.2026.1113296](https://doi.org/10.1016/j.rineng.2026.1113296)

## Overview

This repository provides a simulation-driven machine-learning workflow for predicting dissipated energy in ball-on-flat fretting contacts. The research combines a validated three-dimensional Abaqus finite-element model with supervised machine learning and deploys trained regression models for inference through a Python API.

The broader research investigates two tasks:

1. **Dissipated-energy prediction** using regression models.
2. **Slip-regime classification** into partial slip, mixed slip, and gross slip.

The current repository focuses on data exploration, regression-model development, saved model artifacts, and regression inference through the API application.

## Repository Structure

```text
Fretting-Energy-FEM-ML-Regression-API/
├── main.py
├── explore.py
├── train_regression.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── README.md
├── models/
│   ├── final_dissipated_energy_model.joblib
│   ├── linear_regression_energy.joblib
│   ├── random_forest_energy.joblib
│   ├── svr_energy.joblib
│   └── xgboost_energy.joblib
└── src/
    └── fretting_energy_fem_ml_regression_api/
        └── __init__.py
```

### Main components

- **`main.py`** — Application entry point for regression inference through the API.
- **`explore.py`** — Data inspection and exploratory analysis.
- **`train_regression.py`** — Regression-model training, evaluation, and model serialization.
- **`src/fretting_energy_fem_ml_regression_api/`** — Python package for organizing reusable application functionality.
- **`models/`** — Serialized regression models stored in Joblib format.
- **`pyproject.toml`** — Project metadata, dependencies, and tool configuration.
- **`uv.lock`** — Locked dependency versions for reproducible environments.
- **`.python-version`** — Python version selection for the project.
- **`.gitignore`** — Files and directories excluded from version control.

The saved model files allow trained estimators to be reused without retraining, provided that the corresponding model artifacts are compatible with the inference code and installed dependencies.

## Research Workflow

The overall research workflow consists of the following stages:

1. Define the fretting-contact input parameters and target response.
2. Validate the finite-element model against Hertzian and Cattaneo–Mindlin contact solutions.
3. Generate simulation data using finite-element analysis.
4. Inspect the dataset and investigate input-response relationships.
5. Engineer physically meaningful features.
6. Train and optimize supervised regression models.
7. Evaluate model performance using held-out test data and cross-validation.
8. Serialize the trained models for subsequent inference.
9. Serve dissipated-energy predictions through the API.

## Input Variables and Prediction Target

The study investigates combinations of the following physical parameters:

- Normal load, \(P\)
- Displacement amplitude, \(D\)
- Coefficient of friction, \(\mu\)
- Contact radius, \(a\)

Derived features include:

- \(D/a\)
- \(\mu P\)
- \(\mu PD\)

The regression target is the **dissipated energy per fretting cycle**.

**Data availability:** The published article states that the research data will be made available upon request. Restricted or institutionally controlled datasets should not be redistributed without the appropriate authorization.

## Machine-Learning Models

The research compares several regression algorithms:

- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regression
- XGBoost Regression

The reported XGBoost model with engineered features achieved a held-out test-set \(R^2\) of **0.966** and an RMSE of **0.510**, as reported in the study.

The trained surrogate model can provide predictions in less than 1 ms, compared with approximately 2–6 hours for an individual finite-element simulation under the reported computational conditions.

These performance figures describe the reported experimental setup and may vary with hardware, input conditions, preprocessing, and implementation.

## Installation

### Prerequisites

- Python version compatible with `.python-version` and `pyproject.toml`
- Git
- `uv`
- Dependencies specified in `pyproject.toml` and `uv.lock`

Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 1. Clone the repository

Replace the placeholder URL with your actual GitHub repository URL.

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Fretting-Energy-FEM-ML-Regression-API
```

### 2. Install dependencies

From the repository root, run:

```bash
uv sync
```

This creates or updates the project virtual environment and installs the dependencies declared by the project. When a compatible `uv.lock` is present, `uv` uses it to reproduce the locked dependency versions.

Activate the virtual environment on Windows PowerShell if desired:

```powershell
.venv\Scripts\Activate.ps1
```

Activation is optional when using `uv run`.

## Usage

### 1. Explore the dataset

Run:

```bash
uv run python explore.py
```

The exploration script can be used to inspect the dataset, examine variable distributions, check missing values, and investigate relationships between the input parameters and dissipated energy.

The required dataset must be available at the location expected by the script.

### 2. Train the regression models

Run:

```bash
uv run python train_regression.py
```

The training script is intended to handle model fitting, evaluation, and serialization. Check the implementation for the exact training procedure, dataset path, evaluation metrics, and output filenames.

The serialized model artifacts are stored in the `models/` directory when configured accordingly.

**Note:** Retraining may overwrite existing model files if the script saves to the same paths. Back up any model artifacts you need to preserve before retraining.

### 3. Start the API

If `main.py` contains a FastAPI application instance named `app`, start the development server using:

```bash
uv run uvicorn main:app --reload
```

The server will normally be available at:

- **API root:** http://127.0.0.1:8000
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

These endpoints assume the default Uvicorn host and port and that the application starts successfully.

If Uvicorn is not installed in the project environment, add it using:

```bash
uv add "uvicorn[standard]"
```

### Recommended execution order

```bash
uv sync
uv run python explore.py
uv run python train_regression.py
uv run uvicorn main:app --reload
```

The exploration and training stages are needed when preparing or updating models. If compatible trained model artifacts already exist and the API can load them, you can start the API without retraining.

## Model Artifacts

The repository contains the following serialized models:

| File | Model |
|---|---|
| `linear_regression_energy.joblib` | Linear Regression |
| `random_forest_energy.joblib` | Random Forest Regression |
| `svr_energy.joblib` | Support Vector Regression |
| `xgboost_energy.joblib` | XGBoost Regression |
| `final_dissipated_energy_model.joblib` | Final selected model |

The exact algorithm and preprocessing stored in `final_dissipated_energy_model.joblib` should be confirmed from the training code.

Joblib artifacts should be loaded only from trusted sources. Models serialized with Joblib or pickle-compatible formats can execute code during loading. Compatibility may also depend on the versions of scikit-learn, XGBoost, and other training dependencies.

## Reproducibility

For reproducible model development and deployment:

- Use project-relative paths rather than machine-specific absolute paths.
- Keep `pyproject.toml` and `uv.lock` synchronized with the project dependencies.
- Use consistent random seeds where supported.
- Keep the held-out test set separate from hyperparameter optimization.
- Save the fitted preprocessing pipeline together with the estimator when preprocessing is required.
- Document feature names, units, and expected input ranges.
- Ensure that the API uses the same feature order and preprocessing as training.
- Record the model artifact and dependency versions used for evaluation.
- Avoid committing secrets, virtual environments, caches, and restricted datasets.

## Citation

If this repository or methodology supports your research, please cite the associated publication:

```bibtex
@article{Kumar2026FEMMLFretting,
  title   = {Finite-element and machine-learning framework for dissipated-energy prediction and slip-regime classification in fretting contacts},
  author  = {Kumar, Avneesh and Vostrak, Marek and Houdkova, Sarka},
  journal = {Results in Engineering},
  volume  = {32},
  pages   = {1113296},
  year    = {2026},
  doi     = {10.1016/j.rineng.2026.1113296},
  url     = {https://doi.org/10.1016/j.rineng.2026.1113296}
}
```

## License

The publication's license does not automatically establish the software license for this repository. Consult the authors' and institution's software-release requirements before choosing and adding a separate software license.

## Acknowledgements

This work was created as part of the OP JAK Inter-sectoral Cooperation for ITI project CZ.02.01.01/00/23_021/0008594, *European Laboratory of Applied Research of Additive Technologies for Thermal Spraying*, supported by the European Union.
