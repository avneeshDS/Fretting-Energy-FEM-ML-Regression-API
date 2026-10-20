# Finite-Element and Machine-Learning Framework for Fretting Contacts

[![Paper](https://img.shields.io/badge/Results%20in%20Engineering-2026-005A9C)](https://doi.org/10.1016/j.rineng.2026.1113296)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.rineng.2026.1113296-blue)](https://doi.org/10.1016/j.rineng.2026.1113296)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Abaqus](https://img.shields.io/badge/FEA-Abaqus-005386)](https://www.3ds.com/products/simulia/abaqus)

Code and supporting workflow for the research article:

> **Finite-element and machine-learning framework for dissipated-energy prediction and slip-regime classification in fretting contacts**  
> Avneesh Kumar, Marek Vostrak, and Sarka Houdkova  
> *Results in Engineering*, Volume 32, 2026, Article 1113296  
> <https://doi.org/10.1016/j.rineng.2026.1113296>

## Overview

This repository implements a simulation-driven machine-learning framework for rapid prediction of fretting-contact behaviour in a ball-on-flat configuration. A validated three-dimensional finite-element model is used to generate contact-response data over a design of experiments. Machine-learning models then learn to:

1. predict dissipated energy per fretting cycle; and
2. classify the fretting slip regime as partial slip, mixed slip, or gross slip.

The framework connects physically validated finite-element simulations with interpretable machine learning, enabling new operating conditions to be screened in less than 1 ms after model training.

## Research workflow

The study follows four main stages:

1. **Problem definition**
   - Define the prediction objectives.
   - Select input parameters and output responses.
   - Define the ball-on-flat geometry, contact conditions, and material properties.

2. **Finite-element data generation**
   - Build and mesh the three-dimensional Abaqus model.
   - Validate normal contact against Hertz theory.
   - Validate tangential contact against the Cattaneo-Mindlin partial-slip solution.
   - Perform mesh-sensitivity and contact-response checks.
   - Run the design-of-experiments simulations and extract contact features.

3. **Data preparation and machine learning**
   - Engineer physically meaningful contact features.
   - Scale features where required by the learning algorithm.
   - Use a held-out test set and repeated five-fold cross-validation.
   - Train, optimize, compare, and interpret regression and classification models.

4. **Model outputs**
   - Predict dissipated energy.
   - Classify the slip regime.
   - Evaluate previously unseen fretting operating conditions.

## Dataset and input variables

The machine-learning dataset contains **170 finite-element simulations** covering combinations of:

- normal load, `P`;
- displacement amplitude, `D`;
- coefficient of friction, `mu`;
- ball radius, `a`;
- normalized displacement, `D/a`; and
- normalized friction-capacity features, including `muP` and `muPD`.

The target responses are:

- **dissipated energy per fretting cycle** for regression; and
- **slip regime** for classification:
  - partial slip regime (PSR),
  - mixed slip regime (MSR), and
  - gross slip regime (GSR).

> **Data availability:** The article states that the data will be made available upon request. The repository should not redistribute the dataset unless the authors have explicitly approved public release.

## Finite-element model

The numerical model represents a deformable ball in contact with a flat specimen. The study uses:

- three-dimensional finite-element analysis in Abaqus;
- surface-to-surface contact with finite sliding;
- an augmented-Lagrangian contact formulation;
- a Coulomb friction model;
- quadratic tetrahedral elements for the ball;
- hexahedral elements for the flat specimen; and
- local mesh refinement in the contact region.

The material parameters reported for the ball-on-flat model are:

| Property | Young's modulus | Poisson's ratio |
|---|---:|---:|
| Ball | 380,000 MPa | 0.30 |
| Flat specimen | 208,000 MPa | 0.27 |

The selected contact-region mesh size is **20 micrometres**, based on the mesh-sensitivity study.

## Validation

The finite-element model was validated before generating the machine-learning dataset:

- maximum relative error in normal contact force: **0.43%**;
- maximum relative error in normalized tangential-force ratios: **1.99%**; and
- contact pressure and stick-slip boundaries were checked against Hertz and Cattaneo-Mindlin analytical solutions.

These checks establish that the numerical model reproduces the expected normal and tangential contact mechanics with sufficient accuracy for dataset generation.

## Machine-learning models

### Regression

The evaluated regression models include:

- Linear Regression;
- Support Vector Regression;
- Random Forest Regression; and
- XGBoost Regression.

The models predict dissipated energy from the selected operating parameters and engineered contact features.

### Classification

The evaluated classification models include:

- Logistic Regression;
- Support Vector Classifier;
- Random Forest Classifier; and
- XGBoost Classifier.

The models classify each operating condition into PSR, MSR, or GSR.

### Evaluation strategy

The reported workflow uses:

- a held-out test set;
- repeated five-fold cross-validation with 10 repetitions;
- grid-search hyperparameter optimization;
- regression metrics: R-squared, RMSE, and MAPE; and
- classification metrics: accuracy and macro-F1 score.

Permutation importance and SHAP analysis are used to interpret the trained models and assess the contribution of baseline and derived features.

## Main findings

- XGBoost regression with engineered features achieved a held-out **R-squared of 0.966** and **RMSE of 0.510**.
- The baseline Support Vector Classifier achieved the highest reported mean cross-validation **macro-F1 score of 0.914 +/- 0.115**.
- Feature engineering improved Linear Regression and Logistic Regression more strongly than the nonlinear tree-based models.
- Feature importance was model-dependent. Displacement-related variables, friction capacity, and normalized displacement were among the influential predictors.
- After training, the surrogate models generated predictions in less than 1 ms, compared with approximately 2-6 hours for an individual finite-element simulation.

## Suggested repository structure

```text
.
|-- data/
|   |-- raw/                 # Original simulation data, if distribution is permitted
|   `-- processed/           # Cleaned and feature-engineered datasets
|-- models/                  # Saved fitted models and preprocessing objects
|-- notebooks/               # Optional exploratory or demonstration notebooks
|-- results/
|   |-- figures/             # Generated plots
|   `-- metrics/             # Model-performance summaries
|-- src/
|   |-- explore.py           # Data loading and exploratory analysis
|   |-- features.py          # Feature engineering
|   |-- train_regression.py  # Regression training and optimization
|   |-- train_classifier.py  # Classification training and optimization
|   |-- evaluate.py          # Test-set evaluation and plots
|   `-- predict.py           # Inference for new operating conditions
|-- requirements.txt
|-- LICENSE
`-- README.md
```

Adapt the tree above to the filenames actually included in the repository.

## Installation

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
python -m venv .venv
```

Activate the environment.

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Typical dependencies include `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `shap`, and `joblib`.

## Usage

The exact commands depend on the filenames retained in the repository. A recommended sequence is:

```bash
python src/explore.py
python src/train_regression.py
python src/train_classifier.py
python src/evaluate.py
```

For inference with a saved model:

```bash
python src/predict.py
```

Use project-relative paths or a configuration file rather than local absolute paths such as `V:\\...`, so that the repository runs on other computers.

## Reproducibility recommendations

For reproducible results:

- fix `random_state` consistently;
- save the complete preprocessing-and-model pipeline rather than only the estimator;
- record package versions in `requirements.txt`;
- keep the held-out test set isolated from hyperparameter optimization;
- document feature names, units, and target-label definitions;
- preserve the repeated cross-validation configuration used in the article; and
- avoid committing confidential data, large Abaqus output databases, or proprietary solver files.

## Citation

If this repository or the associated methodology supports your work, please cite:

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

## Related publication

- **Journal:** *Results in Engineering*
- **Volume:** 32
- **Article number:** 1113296
- **Year:** 2026
- **DOI:** <https://doi.org/10.1016/j.rineng.2026.1113296>
- **Open-access license for the article:** CC BY 4.0

## License

The journal article is distributed under the **Creative Commons Attribution 4.0 International license (CC BY 4.0)**. This does not automatically determine the license of the source code in this repository.

Add a separate repository license, such as MIT or BSD-3-Clause, only after confirming the authors' and institution's software-release requirements.

## Contact

For questions about the methodology, data, or collaboration, please contact the corresponding author through the contact information provided in the published article.

## Acknowledgements

The work was created as part of the project of the OP JAK Inter-sectoral Cooperation for ITI CZ.02.01.01/00/23_021/0008594, *European Laboratory of Applied Research of Additive Technologies for Thermal Spraying*, supported by the European Union.
