# Task 2.1: Data Preprocessing & Cleaning
# Comprehensive Heart Disease Data Preprocessing Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load and preprocess the heart disease dataset
    """
    # column names based on UCI Heart Disease dataset documentation
    column_names = [
        'age',        # Age in years
        'sex',        # Sex (1 = male; 0 = female)  
        'cp',         # Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
        'trestbps',   # Resting blood pressure (in mm Hg)
        'chol',       # Serum cholesterol in mg/dl
        'fbs',        # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        'restecg',    # Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)
        'thalach',    # Maximum heart rate achieved
        'exang',      # Exercise induced angina (1 = yes; 0 = no)
        'oldpeak',    # ST depression induced by exercise relative to rest
        'slope',      # Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
        'ca',         # Number of major vessels (0-3) colored by fluoroscopy
        'thal',       # Thallium heart scan (3 = normal; 6 = fixed defect; 7 = reversible defect)
        'target'      # Target variable: 0 = no heart disease, 1-4 = heart disease
    ]
    
    # Cleveland dataset
    print("Loading Heart Disease Dataset...")
    heart_data = pd.read_csv('processed.cleveland.data', header=None, names=column_names)
    
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {heart_data.shape}")
    print(f"\nColumn names and data types:")
    print(heart_data.dtypes)
    
    return heart_data

def handle_missing_values(data):
    """
    Identify and handle missing values in the dataset
    """
    print("\n=== MISSING VALUES ANALYSIS ===")
    print("Missing values in each column:")
    print(data.isnull().sum())
    
    # Check for '?' values which are commonly used as missing value indicators in UCI datasets
    print("\n=== Checking for '?' values ===")
    for col in data.columns:
        if data[col].dtype == 'object':
            unique_vals = data[col].unique()
            print(f"{col}: {unique_vals}")
    
    # Create a copy of the data for cleaning
    data_clean = data.copy()
    
    # Handle '?' values in 'ca' and 'thal' columns
    data_clean['ca'] = data_clean['ca'].replace('?', np.nan)
    data_clean['thal'] = data_clean['thal'].replace('?', np.nan)
    
    # Convert to numeric
    data_clean['ca'] = pd.to_numeric(data_clean['ca'])
    data_clean['thal'] = pd.to_numeric(data_clean['thal'])
    
    print("\nMissing values after identifying '?' as missing:")
    print(data_clean.isnull().sum())
    
    # Impute missing values using mode (most frequent value)
    ca_imputer = SimpleImputer(strategy='most_frequent')
    data_clean['ca'] = ca_imputer.fit_transform(data_clean[['ca']]).ravel()
    
    thal_imputer = SimpleImputer(strategy='most_frequent')  
    data_clean['thal'] = thal_imputer.fit_transform(data_clean[['thal']]).ravel()
    
    print("\nMissing values after imputation:")
    print(data_clean.isnull().sum())
    
    return data_clean

def encode_target_variable(data):
    """
    Convert target variable to binary classification
    """
    print("\n=== TARGET VARIABLE ENCODING ===")
    print("Original target distribution:")
    print(data['target'].value_counts().sort_index())
    
    # Convert target variable to binary (0 = no disease, 1 = disease)
    # Original: 0 = no disease, 1-4 = disease presence (different levels)
    data['target'] = (data['target'] > 0).astype(int)
    
    print("\nTarget variable distribution after conversion to binary:")
    print(data['target'].value_counts())
    
    return data

def scale_features(data):
    """
    Apply feature scaling to numerical features
    """
    print("\n=== FEATURE SCALING ===")
    
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("Features scaled successfully using StandardScaler!")
    print(f"Scaled features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    
    return X_scaled, y, scaler

def save_cleaned_data(data, filename='heart_disease_cleaned.csv'):
    """
    Save the cleaned dataset
    """
    data.to_csv(filename, index=False)
    print(f"\n✅ Cleaned dataset saved as '{filename}'")

def main():
    """
    Main preprocessing pipeline
    """
    print("=== TASK 2.1: DATA PREPROCESSING & CLEANING ===")
    
    # Step 1: Load data
    raw_data = load_and_preprocess_data()
    
    # Step 2: Handle missing values
    clean_data = handle_missing_values(raw_data)
    
    # Step 3: Encode target variable
    clean_data = encode_target_variable(clean_data)
    
    # Step 4: Feature scaling
    X_scaled, y, scaler = scale_features(clean_data)
    
    # Step 5: Save results
    save_cleaned_data(clean_data)
    
    # Step 6: Display summary statistics
    print("\n=== CLEANED DATASET SUMMARY ===")
    print(clean_data.describe())
    
    print(f"\n✅ TASK 2.1 COMPLETE!")
    print(f"Dataset successfully preprocessed and cleaned.")
    print(f"Final dataset shape: {clean_data.shape}")
    
    return clean_data, X_scaled, y, scaler

if __name__ == "__main__":
    cleaned_data, X_scaled, y, scaler = main()