# End-to-End Housing ML Pipeline

A single-script machine learning pipeline that trains a Random Forest model to predict California housing prices, then reuses that same script to run inference on new data — no separate training/inference files needed.

## Overview

This project predicts the median house value of a California census block group from housing and demographic attributes (location, income, room counts, proximity to the ocean, etc.). The same script, `project_housing_price.py`, handles both stages of the ML lifecycle:

- **First run** — no trained model exists yet, so the script trains one: it loads the raw dataset, builds a preprocessing pipeline, fits a `RandomForestRegressor`, and saves both the trained model and the preprocessing pipeline to disk.
- **Every run after that** — a trained model already exists, so the script switches to inference mode: it loads the saved model and pipeline, transforms new input data, and writes predictions to a CSV file.

## Dataset

The project uses the [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) — 20,640 records covering California census block groups from the 1990 census, across 10 columns:

| Column | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates of the block group |
| `housing_median_age` | Median age of houses in the block group |
| `total_rooms`, `total_bedrooms` | Total rooms / bedrooms in the block group |
| `population`, `households` | Population and household counts |
| `median_income` | Median household income (tens of thousands of USD) |
| `ocean_proximity` | Categorical distance-to-ocean label (e.g. `NEAR BAY`, `INLAND`) |
| `median_house_value` | Median house value — **target variable** |

`total_bedrooms` is the only column with missing values, which the pipeline handles automatically via median imputation.

## How It Works

1. **Stratified sampling** — `median_income` is binned into 5 categories, and a `StratifiedShuffleSplit` uses those bins to carve out an 80/20 train/test split, so the test set represents the full income range rather than a random skew.
2. **Held-out set as "new data"** — the 20% test split is saved as `input.csv`, standing in for new, unseen data the trained model will score later.
3. **Preprocessing pipeline** — built with a `ColumnTransformer`:
   - Numeric columns → median imputation + `StandardScaler`
   - `ocean_proximity` → `OneHotEncoder`
4. **Model training** — a `RandomForestRegressor` is fit on the transformed training data; the fitted model and pipeline are serialized with `joblib` as `model.pkl` and `pipeline.pkl`.
5. **Inference** — on the next run, the script detects that `model.pkl` already exists, loads both saved artifacts, transforms `input.csv` through the pipeline, predicts `median_house_value`, and writes the result to `output.csv`.

## Project Structure

```
.
├── project_housing_price.py   # Trains on first run, runs inference on every run after
├── dataset/
│   └── archive/
│       └── housing.csv        # Raw dataset (not included — see Dataset Setup)
├── input.csv                  # Held-out data used for inference (generated on first run)
├── output.csv                 # Predictions (generated on inference runs)
├── pipeline.pkl                # Saved preprocessing pipeline
├── model.pkl                   # Saved trained model (generated on first run)
└── README.md
```

## Getting Started

### Prerequisites
- Python 3
- pandas, numpy, scikit-learn, joblib

### Installation

```bash
git clone https://github.com/mirzaasadijaz/End-to-End-Housing-ML-Pipeline.git
cd End-to-End-Housing-ML-Pipeline
pip install pandas numpy scikit-learn joblib
```

### Dataset Setup

Download the [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) and place `housing.csv` at:

```
dataset/archive/housing.csv
```

### Usage

```bash
python project_housing_price.py
```

- **First run** — trains the model and prints `Model is trained. Congrats!`. This creates `model.pkl`, `pipeline.pkl`, and `input.csv`.
- **Every run after that** — since `model.pkl` now exists, the script runs inference on `input.csv` instead, and prints `Inference is Completed, results saved in output.csv. Enjoy!`.

To re-train from scratch, delete `model.pkl` before running the script again.

## Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — pipelines, `ColumnTransformer`, preprocessing, `RandomForestRegressor`
- **joblib** — model and pipeline persistence

## Possible Improvements

- Add an evaluation step (e.g. RMSE / R² on the test set) — the pipeline currently trains and infers but doesn't report accuracy
- Split training and inference into separate scripts, or add a CLI flag, instead of branching on whether `model.pkl` exists
- Compare Random Forest against other candidate models (Linear Regression, Decision Tree) with cross-validation
- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`

## License

This repository doesn't yet specify a license. Consider adding one (e.g. MIT) to clarify how others can use or build on this code.
