# Finite-Element and Machine-Learning Framework for Fretting Contacts

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.rineng.2026.1113296-blue)](https://doi.org/10.1016/j.rineng.2026.1113296)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Abaqus](https://img.shields.io/badge/FEA-Abaqus-005386)](https://www.3ds.com/products/simulia/abaqus)

Code and API deployment associated with the research article:

> **Finite-element and machine-learning framework for dissipated-energy prediction and slip-regime classification in fretting contacts**  
> Avneesh Kumar, Marek Vostrak, and Sarka Houdkova  
> *Results in Engineering*, Volume 32, 2026, Article 1113296  
> <https://doi.org/10.1016/j.rineng.2026.1113296>

## Overview

This repository provides a simulation-driven machine-learning workflow for predicting dissipated energy in ball-on-flat fretting contacts. The research combines a validated three-dimensional Abaqus finite-element model with supervised machine learning and deploys the trained regression model through a FastAPI application.

The wider study addresses two tasks:

1. dissipated-energy prediction using regression; and
2. slip-regime classification into partial slip, mixed slip, and gross slip.

The files currently included in this repository focus on **data exploration, regression-model development, and regression inference through FastAPI**.

## Repository structure

```text
.
|-- explore.py          # Data loading, inspection, and exploratory analysis
|-- train_regression.py  # Regression training, optimization, evaluation, and model saving
|-- main.py             # FastAPI application for regression predictions
|-- requirements.txt    # Python dependencies
|-- LICENSE             # Software license
`-- README.md           # Project documentation
```

A saved model file may also be required by `main.py`. Add the correct filename to the repository structure after confirming the model filename used in the application.

## Research workflow

1. Define the fretting-contact input parameters and output response.
2. Validate the finite-element model against Hertz and Cattaneo-Mindlin contact solutions.
3. Generate the dataset using 170 finite-element simulations.
4. Inspect the dataset and input-response relationships.
5. Engineer physically meaningful features.
6. Train and optimize the regression models.
7. Evaluate the selected model using held-out test data and repeated cross-validation.
8. Save the fitted model or complete preprocessing pipeline.
9. Serve dissipated-energy predictions through FastAPI.

## Input variables and target

The study investigates combinations of:

- normal load, `P`;
- displacement amplitude, `D`;
- coefficient of friction, `mu`;
- contact radius, `a`; and
- derived features such as `D/a`, `muP`, and `muPD`.

The regression target is the **dissipated energy per fretting cycle**.

> **Data availability:** The published article states that the research data will be made available upon request. Do not upload restricted or institutionally controlled data without approval.

## Model development

The published study compared:

- Linear Regression;
- Support Vector Regression;
- Random Forest Regression; and
- XGBoost Regression.

XGBoost regression with engineered features achieved a held-out test-set **R-squared of 0.966** and **RMSE of 0.510**. The trained surrogate model produced predictions in less than 1 ms, whereas an individual finite-element simulation required approximately 2 to 6 hours.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Explore the dataset

```bash
python explore.py
```

`explore.py` is intended for data loading, structural inspection, descriptive analysis, missing-value checks, and exploration of the model inputs and target response.

### 2. Train and save the regression model

```bash
python trainregression.py
```

`trainregression.py` should contain the preprocessing pipeline, cross-validation strategy, hyperparameter optimization, final evaluation, and model serialization. Saving the complete fitted pipeline is recommended because preprocessing will then be applied consistently during API inference.

### 3. Run the FastAPI application

```bash
uvicorn main:app --reload
```

The command assumes that `main.py` contains a FastAPI instance named `app`.

After starting the server, use:

- API root: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>

### Recommended execution order

```bash
python explore.py
python trainregression.py
uvicorn main:app --reload
```

## Reproducibility

For reproducible model development:

- use project-relative paths instead of computer-specific paths such as `V:\...`;
- define `random_state` consistently;
- keep the held-out test set separate from hyperparameter optimization;
- save the fitted preprocessing-and-regression pipeline;
- document all feature names and units;
- record package versions in `requirements.txt`; and
- ensure that `main.py` loads the same model artifact created by `trainregression.py`.

## Citation

If this repository or methodology supports your research, please cite:

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

The published article is available under the Creative Commons Attribution 4.0 International license. That publication license does not automatically define the license for the source code in this repository. Add a separate software license only after checking the authors' and institution's software-release requirements.

## Acknowledgements

This work was created as part of the OP JAK Inter-sectoral Cooperation for ITI project CZ.02.01.01/00/23_021/0008594, *European Laboratory of Applied Research of Additive Technologies for Thermal Spraying*, supported by the European Union.
