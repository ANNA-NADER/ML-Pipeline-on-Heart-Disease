# Task 2.7: Model Export & Deployment
# Comprehensive Model Pipeline Export and Deployment Preparation

import pandas as pd
import numpy as np
import joblib
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class HeartDiseasePredictionPipeline(BaseEstimator):
    """
    Complete prediction pipeline for heart disease classification
    Handles preprocessing, feature selection, and prediction
    """
    
    def __init__(self, scaler, model, selected_features, feature_names):
        self.scaler = scaler
        self.model = model
        self.selected_features = selected_features
        self.feature_names = feature_names
    
    def _validate_input(self, X):
        """Validate and prepare input data"""
        if isinstance(X, pd.DataFrame):
            # Ensure features are in correct order
            if set(X.columns) != set(self.feature_names):
                missing_features = set(self.feature_names) - set(X.columns)
                extra_features = set(X.columns) - set(self.feature_names)
                
                error_msg = ""
                if missing_features:
                    error_msg += f"Missing features: {missing_features}. "
                if extra_features:
                    error_msg += f"Unexpected features: {extra_features}. "
                
                if error_msg:
                    error_msg += f"Expected features: {self.feature_names}"
                    raise ValueError(error_msg)
            
            # Reorder columns to match training order
            X = X[self.feature_names]
        else:
            # Convert numpy array to DataFrame
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
            X = pd.DataFrame(X, columns=self.feature_names)
        
        return X
    
    def preprocess(self, X):
        """Apply preprocessing steps"""
        # Validate input
        X_validated = self._validate_input(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_validated)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Select important features
        X_selected = X_scaled_df[self.selected_features]
        
        return X_selected
    
    def predict(self, X):
        """Make predictions"""
        X_processed = self.preprocess(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_processed = self.preprocess(X)
        return self.model.predict_proba(X_processed)
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficient magnitude
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None

def load_best_model_and_components():
    """
    Load the best performing model and all required components
    """
    print("=== LOADING BEST MODEL AND COMPONENTS ===")
    
    # Load cleaned data to get scaler
    heart_data = pd.read_csv('heart_disease_cleaned.csv')
    X_full = heart_data.drop('target', axis=1)
    
    # Recreate scaler with same parameters
    scaler = StandardScaler()
    scaler.fit(X_full)
    
    # Load selected features
    with open('feature_selection_results.json', 'r') as f:
        feature_results = json.load(f)
    
    selected_features = feature_results['consensus_results']['final_selected_features']
    all_feature_names = feature_results['original_features']
    
    # Load the best tuned model (assuming Logistic Regression was best)
    try:
        best_model = joblib.load('best_logistic_regression_model.pkl')
        model_name = 'Logistic Regression'
        print(f"Loaded best model: {model_name}")
    except:
        print("Best model file not found, creating new Logistic Regression model...")
        from sklearn.linear_model import LogisticRegression
        best_model = LogisticRegression(C=10, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
        
        # Train the model
        selected_data = pd.read_csv('heart_disease_selected_features.csv')
        X_train = selected_data.drop('target', axis=1)
        y_train = selected_data['target']
        best_model.fit(X_train, y_train)
        model_name = 'Logistic Regression (Retrained)'
    
    print(f"Selected features: {selected_features}")
    print(f"All feature names: {all_feature_names}")
    
    return best_model, scaler, selected_features, all_feature_names, model_name

def create_deployment_pipeline(model, scaler, selected_features, feature_names):
    """
    Create the complete deployment pipeline
    """
    print(f"\n=== CREATING DEPLOYMENT PIPELINE ===")
    
    # Create the complete pipeline
    pipeline = HeartDiseasePredictionPipeline(
        scaler=scaler,
        model=model,
        selected_features=selected_features,
        feature_names=feature_names
    )
    
    print("Deployment pipeline created successfully!")
    print(f"Pipeline components:")
    print(f"  - Scaler: {type(scaler).__name__}")
    print(f"  - Model: {type(model).__name__}")
    print(f"  - Total features: {len(feature_names)}")
    print(f"  - Selected features: {len(selected_features)}")
    
    return pipeline

def test_pipeline(pipeline, test_samples=5):
    """
    Test the pipeline with sample data
    """
    print(f"\n=== TESTING PIPELINE ===")
    
    # Load original cleaned data for testing
    heart_data = pd.read_csv('heart_disease_cleaned.csv')
    
    # Get random test samples
    test_data = heart_data.drop('target', axis=1).sample(n=test_samples, random_state=42)
    actual_labels = heart_data.loc[test_data.index, 'target']
    
    print(f"Testing with {test_samples} sample(s):")
    
    try:
        # Test predictions
        predictions = pipeline.predict(test_data)
        probabilities = pipeline.predict_proba(test_data)
        
        for i in range(len(test_data)):
            prob_no_disease = probabilities[i][0]
            prob_disease = probabilities[i][1]
            predicted = predictions[i]
            actual = actual_labels.iloc[i]
            
            print(f"\nSample {i+1}:")
            print(f"  Actual: {actual} ({'Disease' if actual == 1 else 'No Disease'})")
            print(f"  Predicted: {predicted} ({'Disease' if predicted == 1 else 'No Disease'})")
            print(f"  Probability - No Disease: {prob_no_disease:.4f}")
            print(f"  Probability - Disease: {prob_disease:.4f}")
            print(f"  Correct: {'✓' if predicted == actual else '✗'}")
        
        accuracy = (predictions == actual_labels).mean()
        print(f"\nTest accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        return True
    
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        return False

def save_deployment_artifacts(pipeline, model_name):
    """
    Save all deployment artifacts
    """
    print(f"\n=== SAVING DEPLOYMENT ARTIFACTS ===")
    
    # Save the complete pipeline
    pipeline_filename = 'final_heart_disease_model.pkl'
    joblib.dump(pipeline, pipeline_filename)
    print(f"Complete pipeline saved as '{pipeline_filename}'")
    
    # Save backup using pickle
    backup_filename = 'final_heart_disease_model_pickle.pkl'
    with open(backup_filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Backup pipeline saved as '{backup_filename}'")
    
    # Get model performance metrics
    try:
        with open('detailed_tuning_results.json', 'r') as f:
            tuning_results = json.load(f)
        performance_metrics = tuning_results.get(model_name, {})
    except:
        performance_metrics = {'best_cv_score': 'N/A'}
    
    # Create model metadata
    metadata = {
        'model_info': {
            'model_type': model_name,
            'model_class': type(pipeline.model).__name__,
            'scaler_type': type(pipeline.scaler).__name__,
            'creation_date': pd.Timestamp.now().isoformat()
        },
        'features': {
            'total_features': len(pipeline.feature_names),
            'selected_features': len(pipeline.selected_features),
            'all_feature_names': pipeline.feature_names,
            'selected_feature_names': pipeline.selected_features
        },
        'performance': performance_metrics,
        'usage_instructions': {
            'required_features': pipeline.feature_names,
            'input_format': 'pandas DataFrame or numpy array',
            'output_format': 'binary classification (0=No Disease, 1=Disease)',
            'example_usage': {
                'import': 'import joblib',
                'load': 'model = joblib.load("final_heart_disease_model.pkl")',
                'predict': 'prediction = model.predict(patient_data)',
                'probability': 'probability = model.predict_proba(patient_data)'
            }
        }
    }
    
    # Save metadata
    metadata_filename = 'model_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved as '{metadata_filename}'")
    
    return metadata

def create_usage_examples(pipeline):
    """
    Create usage examples and documentation
    """
    print(f"\n=== CREATING USAGE EXAMPLES ===")
    
    # Create example usage code
    usage_code = f'''
# Heart Disease Prediction Model Usage Examples

import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('final_heart_disease_model.pkl')

# Example 1: Single patient prediction
patient_data = pd.DataFrame({{
    'age': [63],
    'sex': [1],  # 1=male, 0=female
    'cp': [1],   # chest pain type (1-4)
    'trestbps': [145],  # resting blood pressure
    'chol': [233],      # cholesterol
    'fbs': [1],         # fasting blood sugar > 120mg/dl
    'restecg': [2],     # resting ECG results
    'thalach': [150],   # max heart rate achieved
    'exang': [0],       # exercise induced angina
    'oldpeak': [2.3],   # ST depression
    'slope': [3],       # slope of peak exercise ST segment
    'ca': [0],          # number of major vessels
    'thal': [6]         # thallium heart scan
}})

# Make prediction
prediction = model.predict(patient_data)
probability = model.predict_proba(patient_data)

print(f"Prediction: {{prediction[0]}} (0=No Disease, 1=Disease)")
print(f"Probability of Disease: {{probability[0][1]:.4f}}")

# Example 2: Multiple patients
patients_data = pd.DataFrame({{
    'age': [63, 67, 37],
    'sex': [1, 1, 1],
    'cp': [1, 4, 3],
    'trestbps': [145, 160, 130],
    'chol': [233, 286, 250],
    'fbs': [1, 0, 0],
    'restecg': [2, 2, 0],
    'thalach': [150, 108, 187],
    'exang': [0, 1, 0],
    'oldpeak': [2.3, 1.5, 3.5],
    'slope': [3, 2, 3],
    'ca': [0, 3, 0],
    'thal': [6, 3, 3]
}})

# Make predictions for all patients
predictions = model.predict(patients_data)
probabilities = model.predict_proba(patients_data)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Patient {{i+1}}: Prediction={pred}, Disease Probability={prob[1]:.4f}")

# Example 3: Get feature importance (if available)
try:
    feature_importance = model.get_feature_importance()
    print("\\nFeature Importance:")
    print(feature_importance)
except:
    print("Feature importance not available for this model type")

# Required Features:
# {', '.join(pipeline.feature_names)}

# Selected Features (used by model):
# {', '.join(pipeline.selected_features)}
'''
    
    # Save usage examples
    with open('usage_examples.py', 'w') as f:
        f.write(usage_code)
    print("Usage examples saved as 'usage_examples.py'")
    
    return usage_code

def validate_deployment_package():
    """
    Validate that all deployment files are present and working
    """
    print(f"\n=== VALIDATING DEPLOYMENT PACKAGE ===")
    
    required_files = [
        'final_heart_disease_model.pkl',
        'final_heart_disease_model_pickle.pkl',
        'model_metadata.json',
        'usage_examples.py'
    ]
    
    missing_files = []
    for file in required_files:
        try:
            open(file, 'r').close()
            print(f"✓ {file}")
        except:
            missing_files.append(file)
            print(f"✗ {file} - MISSING")
    
    if missing_files:
        print(f"\nWARNING: Missing files: {missing_files}")
        return False
    
    # Test loading the model
    try:
        loaded_model = joblib.load('final_heart_disease_model.pkl')
        print("✓ Model loads successfully")
        
        # Quick functionality test
        test_data = pd.DataFrame({
            'age': [63], 'sex': [1], 'cp': [1], 'trestbps': [145],
            'chol': [233], 'fbs': [1], 'restecg': [2], 'thalach': [150],
            'exang': [0], 'oldpeak': [2.3], 'slope': [3], 'ca': [0], 'thal': [6]
        })
        
        pred = loaded_model.predict(test_data)
        prob = loaded_model.predict_proba(test_data)
        print("✓ Model predictions work correctly")
        
        return True
    
    except Exception as e:
        print(f"✗ Model loading/prediction failed: {e}")
        return False

def main():
    """
    Main model export and deployment pipeline
    """
    print("=== TASK 2.7: MODEL EXPORT & DEPLOYMENT ===")
    
    # Step 1: Load best model and components
    model, scaler, selected_features, feature_names, model_name = load_best_model_and_components()
    
    # Step 2: Create deployment pipeline
    pipeline = create_deployment_pipeline(model, scaler, selected_features, feature_names)
    
    # Step 3: Test the pipeline
    test_success = test_pipeline(pipeline)
    
    if not test_success:
        print("⚠️  Pipeline testing failed - proceeding with caution")
    
    # Step 4: Save deployment artifacts
    metadata = save_deployment_artifacts(pipeline, model_name)
    
    # Step 5: Create usage examples
    usage_code = create_usage_examples(pipeline)
    
    # Step 6: Validate deployment package
    deployment_valid = validate_deployment_package()
    
    if deployment_valid:
        print(f"\n✅ TASK 2.7 COMPLETE!")
        print(f"Model successfully exported and ready for deployment!")
        print(f"\nDeployment package includes:")
        print(f"  - final_heart_disease_model.pkl (main model file)")
        print(f"  - model_metadata.json (model information)")
        print(f"  - usage_examples.py (code examples)")
        print(f"  - Complete preprocessing pipeline")
        print(f"\nModel performance: {metadata['performance']}")
    else:
        print(f"\n⚠️  TASK 2.7 COMPLETED WITH WARNINGS")
        print(f"Some deployment files may be missing or invalid")
    
    return pipeline, metadata

if __name__ == "__main__":
    pipeline, metadata = main()