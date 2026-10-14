# AQI Prediction using Random Forest Regressor

Predicts the **next day's Air Quality Index (AQI)** of an air-quality monitoring station from the current day's pollutant concentrations, using a **Random Forest Regressor**.

## Repository Contents

| File | Description |
|---|---|
| `AQI_Prediction_Random_Forest.ipynb` | Jupyter notebook with the complete step-by-step ML pipeline |
| `random_forest_aqi_model.pkl` | Trained Random Forest model (saved with joblib) |
| `aqi_imputer.pkl` | Median imputer used to fill missing pollutant values |
| `requirements.txt` | Python libraries required |

## Dataset

`station_day.csv` from the Kaggle dataset **Air Quality Data in India (2015–2020)** — daily readings from 110 monitoring stations (108,035 rows).

- **Input features (12):** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target:** `Future_AQI` — the next day's AQI for the same station (created by shifting `AQI` by one day per station)

> The dataset is not included in this repository. Download `station_day.csv` and place it in the same folder as the notebook.

## ML Pipeline

1. Import libraries
2. Load the dataset
3. Understand the data
4. Handling missing values (imputation)
5. Data preprocessing & target creation
6. Feature selection
7. Train–test split
8. Model building – Random Forest Regressor
9. Prediction
10. Model evaluation
11. Feature importance
12. Saving the model (Serialization)
13. Loading the model & predicting on new data (Reusability)
14. Conclusion

## Model

| Hyperparameter | Value |
|---|---|
| Algorithm | RandomForestRegressor |
| `n_estimators` | 50 |
| `max_depth` | 10 |
| `random_state` | 42 |
| Train / Test split | 80% / 20% |
| Missing value handling | Median imputation |

## Results

| Metric | Value |
|---|---|
| MSE | 2172.84 |
| RMSE | 46.61 |
| MAE | 28.32 |
| R² (test) | 0.878 |
| R² (train) | 0.914 |

## How to Run

```bash
git clone https://github.com/Kumar-2006/AQI_Prediction.git
cd AQI_Prediction
pip install -r requirements.txt
jupyter notebook AQI_Prediction_Random_Forest.ipynb
```

## Using the Saved Model

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load('random_forest_aqi_model.pkl')
imputer = joblib.load('aqi_imputer.pkl')

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
today = pd.DataFrame([[81.4, 124.5, 1.44, 20.5, 12.08, 10.72, 0.12, 15.24, 127.09, 0.2, 6.5, np.nan]], columns=features)

print('Predicted next-day AQI:', model.predict(imputer.transform(today))[0])
```

## AQI Categories

| AQI | Category |
|---|---|
| 0 – 50 | Good |
| 51 – 100 | Satisfactory |
| 101 – 200 | Moderate |
| 201 – 300 | Poor |
| 301 – 400 | Very Poor |
| 401 – 500 | Severe |

## Limitations & Future Work

- A random train–test split is used; a time-based split would give a more realistic estimate for forecasting.
- No weather features (wind speed, rainfall, temperature, humidity), which strongly affect next-day AQI.
- Adding previous days' AQI, station/city and season as features could improve accuracy.
