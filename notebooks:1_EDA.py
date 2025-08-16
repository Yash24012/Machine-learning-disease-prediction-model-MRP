import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Load datasets
diabetes = pd.read_csv('data/diabetes.csv')
heart = pd.read_csv('data/heart.csv')

# Diabetes EDA
def eda_diabetes(df):
    # Handle zeros as missing values
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, pd.NA)
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    df[cols] = imputer.fit_transform(df[cols])
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, col in enumerate(cols + ['Age']):
        sns.histplot(df[col], kde=True, ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f'{col} Distribution')
    plt.tight_layout()
    plt.savefig('diabetes_distributions.png')
    
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig('diabetes_correlation.png')
    
    # Handle imbalance
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

# Similar function for heart disease
# ...

# Execute EDA
diabetes_balanced = eda_diabetes(diabetes)
heart_balanced = eda_heart(heart)