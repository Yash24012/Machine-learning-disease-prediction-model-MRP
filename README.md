# Healthcare Disease Prediction System

Predicts diabetes and cardiovascular disease risk using machine learning with explainable AI.

## Features
- Diabetes prediction (Pima Indians dataset)
- Heart disease prediction (Cleveland dataset)
- SHAP explainability for predictions
- Streamlit web interface

## Installation
```bash
git clone https://github.com/yourusername/healthcare-disease-prediction.git
cd healthcare-disease-prediction
pip install -r requirements.txt
```

## Usage
1. Run EDA and model training:
```bash
jupyter notebook notebooks/1_EDA.ipynb
jupyter notebook notebooks/2_Model_Training.ipynb
```

2. Start the Streamlit app:
```bash
streamlit run src/app.py
```

## Datasets
- Diabetes: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- Heart Disease: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## Model Performance
| Model              | Diabetes (ROC AUC) | Heart Disease (ROC AUC) |
|--------------------|--------------------|-------------------------|
| Logistic Regression| 0.85               | 0.93                    |
| Random Forest      | 0.88               | 0.95                    |
| XGBoost            | 0.89               | 0.96                    |
