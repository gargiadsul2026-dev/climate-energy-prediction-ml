# climate-energy-prediction-ml
Comparing SVR and 1D CNN deep learning models to predict CO₂ emissions and temperature trends on an 18,720-row climate and energy dataset.
# climate-energy-prediction-ml

Predicting CO₂ emissions and average temperature trends from real-world climate and energy data, comparing a traditional machine learning approach (Support Vector Regression) against a deep learning approach (1D Convolutional Neural Network).

## Overview

This project builds and evaluates two types of models to predict:
1. **CO₂ emissions** (tons/day)
2. **Average temperature** (°C)

using a climate and energy dataset of **18,720 rows** across multiple countries, with features including humidity, energy consumption, renewable energy share, urban population, industrial activity, and energy pricing.

## Models

### Support Vector Regression (SVR)
- Kernel: RBF
- C = 100, epsilon = 0.1
- Trained on a stratified sample (500 rows per country) with standardized features and target

### 1D Convolutional Neural Network (CNN)
- 30-day sliding window (lookback) to capture temporal patterns
- Architecture: 3 convolutional blocks (Conv1D → BatchNorm → MaxPooling) followed by a dense head with dropout for regularisation
- Trained with early stopping and learning rate reduction on plateau

## Methodology

- Data cleaning and preprocessing (date parsing, feature engineering: month, day of year, year)
- One-hot encoding of country as a categorical feature
- Feature scaling with `StandardScaler`
- Train/test split with fixed random state for reproducibility
- Model evaluation using **MAE**, **RMSE**, and **R² score**

## Results

| Model | Target | MAE | RMSE | R² |
|---|---|---|---|---|
| SVR | CO₂ Emissions (tons/day) | 234.58 | 294.39 | -0.63 |
| SVR | Avg Temperature (°C) | 1.84 | 2.33 | 0.95 |
| 1D CNN | CO₂ Emissions (tons/day) | 199.60 | 234.13 | 0.02 |
| 1D CNN | Avg Temperature (°C) | 6.59 | 7.72 | 0.32 |

Temperature prediction was considerably more successful than CO₂ emissions prediction for both models. SVR achieved strong performance on temperature (R² = 0.95), while both models struggled to capture the variance in CO₂ emissions — likely due to the feature set not fully capturing the drivers of emissions variability, or the need for further feature engineering and hyperparameter tuning. This highlights an area for further investigation rather than a finished result.

## Tech Stack

- Python
- Pandas, NumPy — data manipulation
- Scikit-learn — SVR, preprocessing, evaluation metrics
- TensorFlow / Keras — 1D CNN architecture
- Matplotlib — visualisation

## Project Structure
climate-energy-prediction-ml/
├── README.md
├── requirements.txt
└── SVR_1D_CNN_Predictions.ipynb

## Setup

```bash
pip install -r requirements.txt
```

Then open `SVR_1D_CNN_Predictions.ipynb` in Jupyter or your preferred notebook environment.
the dataset isn't included in the repo.
## Author

Gargi Adsul — MSc Artificial Intelligence, Brunel University of London
