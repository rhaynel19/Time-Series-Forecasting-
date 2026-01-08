Time Series Sales Forecasting — Data Science Portfolio Project
Project Overview

This project focuses on time series forecasting to predict future sales values using historical data and Machine Learning models.
The workflow follows an end-to-end data science pipeline, including data understanding, cleaning, feature engineering, model training, evaluation, and hyperparameter tuning.

The goal is to compare baseline regression models with ensemble-based approaches and select the most accurate model for forecasting.

🎯 Project Objectives

Analyze historical sales data and temporal patterns

Clean and prepare time series data for modeling

Engineer relevant features for forecasting

Train and evaluate multiple regression models

Optimize model performance using hyperparameter tuning

Select and persist the best-performing model

🧠 Models Implemented

Linear Regression

Random Forest Regressor

📊 Evaluation Metrics

Models were evaluated using standard regression metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

📈 Model Performance
🔹 Random Forest Regressor (Test Set)
Metric	Value
MAE	0.0147
RMSE	0.0325
R²	0.9990

🔹 Linear Regression (Test Set)
Metric	Value
MAE	0.1231
RMSE	0.1725
R²	0.9721

🔹 Random Forest Regressor (Training Set)
Metric	Value
MAE	0.0059
RMSE	0.0201
R²	0.9996

🧪 Model Comparison Summary
Model	MAE	RMSE	R²
Random Forest Regressor	0.0147	0.0325	0.9990
Linear Regression	0.1231	0.1725	0.9721

➡️ Best model: Random Forest Regressor

⚙️ Hyperparameter Tuning

GridSearchCV

3-fold cross-validation

36 parameter combinations

108 total fits

Best Hyperparameters
{
  'max_depth': None,
  'min_samples_split': 2,
  'n_estimators': 300
}

📁 Project Structure (ACTUAL)
time-series-forecasting/
│── .venv/                          # Virtual environment (ignored in Git)
│── understanding_data.ipynb        # Data understanding and initial analysis
│── cleaning_data.ipynb             # Data cleaning and preprocessing
│── eda_data.ipynb                  # Exploratory Data Analysis
│── features_data.ipynb             # Feature engineering
│── split_train_randomforest_logreg_tuning_eval.ipynb
│                                   # Train-test split, modeling, tuning, evaluation
│── test_model.ipynb                # Model testing and validation
│── sales_data_sample.csv           # Raw sample dataset
│── sales_data_cleaned.csv           # Cleaned dataset
│── sales_data_features2.csv         # Feature-engineered dataset
│── linear_regression_model.pkl      # Trained Linear Regression model
│── random_forest_model.pkl          # Trained Random Forest model
│── README.md                        # Project documentation

📦 Saved Models

The trained models are stored for reuse:

linear_regression_model.pkl

random_forest_model.pkl

Example: Load and Predict
import joblib
import pandas as pd

model = joblib.load("random_forest_model.pkl")

sample_data = pd.DataFrame({
    "feature_1": [100],
    "feature_2": [30],
    "feature_3": [7]
})

prediction = model.predict(sample_data)
print("Forecast:", prediction)

📌 Key Insights

Random Forest captures non-linear temporal patterns more effectively

Linear Regression serves as a strong baseline but underperforms in complex trends

Very low MAE and RMSE indicate minimal forecasting error

Slight difference between training and test scores suggests good generalization

🚀 Future Improvements

Add lag and rolling window features

Use time-aware validation (TimeSeriesSplit)

Compare with ARIMA, SARIMA, and Prophet

Deploy the model via API or dashboard

📝 Conclusion

This project demonstrates how Machine Learning models, particularly ensemble methods, can achieve highly accurate results in time series forecasting tasks.
The Random Forest Regressor proved to be the most reliable model for predicting sales trends based on historical data.

👨‍💻 Author

Fraimel (Rhayner)
Data Analyst | Data Scientist | Machine Learning Enthusiast

📬 Open to freelance projects, collaborations, and professional opportunities.
