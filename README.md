# ML-Pipeline-on-Heart-Disease
Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset Project

# Heart Disease Prediction - Machine Learning Pipeline

## Project Overview
This project implements a comprehensive machine learning pipeline for heart disease prediction using the UCI Heart Disease dataset. The pipeline includes data preprocessing, feature selection, dimensionality reduction, multiple classification algorithms, clustering analysis, and hyperparameter optimization.

## Dataset Information
- **Source**: UCI Heart Disease Dataset (Cleveland database)
- **Total Samples**: 303
- **Original Features**: 13
- **Target Distribution**: 
  - No Disease: 164 patients
  - Disease: 139 patients

## Pipeline Steps Completed

### 1. Data Preprocessing & Cleaning 
- Loaded Heart Disease UCI dataset
- Handled missing values (ca: 4 missing, thal: 2 missing) using mode imputation
- Converted target variable to binary (0=no disease, 1=disease)
- Applied StandardScaler for feature normalization

### 2. Dimensionality Reduction (PCA) 
- Applied Principal Component Analysis
- Determined optimal components for 95% variance retention: 12 components
- Variance retained: 0.9728 (97.28%)

### 3. Feature Selection 
- Used multiple feature selection methods:
  - Random Forest Feature Importance
  - Recursive Feature Elimination (RFE)
  - ANOVA F-test
- Final selected features (8 features): sex, cp, trestbps, thalach, exang, oldpeak, ca, thal

### 4. Supervised Learning 
Trained and evaluated 4 classification models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### 5. Unsupervised Learning 
- K-Means Clustering (optimal clusters: 2)
- Hierarchical Clustering
- Cluster evaluation using Adjusted Rand Index and Silhouette Score

### 6. Hyperparameter Tuning 
- Applied GridSearchCV and RandomizedSearchCV
- Optimized all models for best performance
- Best performing model: Logistic Regression

### 7. Model Export 
- Saved best model as final_heart_disease_model.pkl
- Created complete prediction pipeline with preprocessing
- Model performance:
  - Accuracy: 0.8689
  - F1-Score: 0.8667
  - AUC Score: 0.9459

## Files Generated

### Data Files
- `heart_disease_cleaned.csv` - Cleaned and preprocessed dataset
- `heart_disease_pca.csv` - PCA-transformed dataset
- `heart_disease_selected_features.csv` - Dataset with selected features only

### Model Files
- `final_heart_disease_model.pkl` - Final trained model (primary)
- `final_heart_disease_model_pickle.pkl` - Final trained model (backup)
- `model_metadata.json` - Model configuration and performance metrics

### Results Files
- `model_performance_results.csv` - Performance comparison of all models
- `clustering_results.csv` - Clustering analysis results
- `hyperparameter_tuning_results.csv` - Hyperparameter tuning comparison
- `final_project_report.json` - Comprehensive project summary

### Visualization Data
- `pca_variance_data.json` - Data for PCA variance charts
- `feature_selection_results.json` - Feature importance and selection data
- `roc_curve_data.json` - ROC curve data for all models
- `elbow_curve_data.json` - K-means optimization data

## Usage

### Loading and Using the Trained Model
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('final_heart_disease_model.pkl')

# Prepare sample data (ensure all 13 features are present)
sample_data = pd.DataFrame([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]], 
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Make predictions
prediction = model.predict(sample_data)
probability = model.predict_proba(sample_data)

print(f"Prediction: {prediction[0]} (0=No Disease, 1=Disease)")
print(f"Probability: {probability[0][1]:.4f} chance of heart disease")
```

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Model Performance Summary
The final Logistic Regression model achieved excellent performance on the test set:
- **Accuracy**: 0.8689 (86.89%)
- **F1-Score**: 0.8667 (86.67%)
- **AUC Score**: 0.9459 (94.59%)

This indicates strong predictive capability for heart disease classification.

## Conclusion
This comprehensive machine learning pipeline successfully demonstrates the complete workflow for heart disease prediction, from data preprocessing to model deployment. The final model shows excellent performance and is ready for practical use in heart disease risk assessment.
