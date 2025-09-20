# Task 2.4: Supervised Learning - Classification Models
# Comprehensive Classification Pipeline with Multiple Algorithms

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve)
import json

def load_selected_features_data():
    """
    Load the dataset with selected features from Task 2.3
    """
    print("=== LOADING SELECTED FEATURES DATA ===")
    
    # Load dataset with selected features
    selected_data = pd.read_csv('heart_disease_selected_features.csv')
    
    # Separate features and target
    X = selected_data.drop('target', axis=1)
    y = selected_data['target']
    
    print(f"Data loaded successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Selected features: {list(X.columns)}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification
    """
    print(f"\n=== SPLITTING DATA ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"Features: {X_train.shape[1]}")
    
    # Check class distribution in splits
    print(f"\nTraining set target distribution:")
    print(y_train.value_counts(normalize=True))
    print(f"Test set target distribution:")
    print(y_test.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test

def initialize_models():
    """
    Initialize all classification models
    """
    print(f"\n=== INITIALIZING MODELS ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    print(f"Initialized {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
    
    return models

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a single model
    """
    print(f"\n--- Training {model_name} ---")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC Score: {auc_score:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {model_name}:")
    print(f"              Predicted")
    print(f"Actual    0    1")
    print(f"    0   {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"    1   {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }

def cross_validate_models(models, X, y, cv_folds=5):
    """
    Perform cross-validation for all models
    """
    print(f"\n=== CROSS-VALIDATION ANALYSIS ({cv_folds}-fold) ===")
    
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"\nCross-validating {model_name}...")
        
        # Perform cross-validation with multiple scoring metrics
        cv_accuracy = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_precision = cross_val_score(model, X, y, cv=cv_folds, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=cv_folds, scoring='recall')
        cv_f1 = cross_val_score(model, X, y, cv=cv_folds, scoring='f1')
        cv_roc_auc = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
        
        cv_results[model_name] = {
            'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std()},
            'precision': {'mean': cv_precision.mean(), 'std': cv_precision.std()},
            'recall': {'mean': cv_recall.mean(), 'std': cv_recall.std()},
            'f1_score': {'mean': cv_f1.mean(), 'std': cv_f1.std()},
            'roc_auc': {'mean': cv_roc_auc.mean(), 'std': cv_roc_auc.std()}
        }
        
        print(f"  Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"  Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
        print(f"  Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
        print(f"  F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        print(f"  ROC-AUC:   {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")
    
    return cv_results

def create_results_summary(model_results):
    """
    Create comprehensive results summary
    """
    print(f"\n=== MODEL PERFORMANCE SUMMARY ===")
    
    # Create summary DataFrame
    summary_data = {}
    for model_name, results in model_results.items():
        summary_data[model_name] = {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'AUC Score': results['auc_score']
        }
    
    results_df = pd.DataFrame(summary_data).round(4)
    print(results_df)
    
    # Find best performing model
    best_model_f1 = max(model_results.keys(), key=lambda k: model_results[k]['f1_score'])
    best_model_auc = max(model_results.keys(), key=lambda k: model_results[k]['auc_score'])
    
    print(f"\nBest F1-Score: {best_model_f1} ({model_results[best_model_f1]['f1_score']:.4f})")
    print(f"Best AUC Score: {best_model_auc} ({model_results[best_model_auc]['auc_score']:.4f})")
    
    return results_df, best_model_f1, best_model_auc

def save_model_results(model_results, cv_results, results_df):
    """
    Save all model results and analysis
    """
    print(f"\n=== SAVING MODEL RESULTS ===")
    
    # Save performance summary
    results_df.to_csv('model_performance_results.csv')
    print("Performance summary saved to 'model_performance_results.csv'")
    
    # Prepare ROC curve data for visualization
    roc_data = {}
    for model_name, results in model_results.items():
        roc_data[model_name] = {
            'fpr': results['roc_curve']['fpr'],
            'tpr': results['roc_curve']['tpr'],
            'auc': results['auc_score']
        }
    
    with open('roc_curve_data.json', 'w') as f:
        json.dump(roc_data, f)
    print("ROC curve data saved to 'roc_curve_data.json'")
    
    # Save detailed results
    detailed_results = {}
    for model_name, results in model_results.items():
        detailed_results[model_name] = {
            'test_performance': {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'auc_score': results['auc_score']
            },
            'cross_validation': cv_results.get(model_name, {}),
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
    
    with open('detailed_model_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print("Detailed results saved to 'detailed_model_results.json'")

def analyze_feature_importance(model_results, feature_names):
    """
    Analyze feature importance for tree-based models
    """
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    importance_results = {}
    
    for model_name, results in model_results.items():
        model = results['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} Feature Importance:")
            for idx, row in feature_imp_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            importance_results[model_name] = feature_imp_df
        
        elif hasattr(model, 'coef_'):
            # For linear models, use coefficient magnitude
            coefficients = np.abs(model.coef_[0])
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': coefficients
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} Feature Coefficients (absolute):")
            for idx, row in feature_imp_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            importance_results[model_name] = feature_imp_df
    
    return importance_results

def main():
    """
    Main supervised learning pipeline
    """
    print("=== TASK 2.4: SUPERVISED LEARNING - CLASSIFICATION MODELS ===")
    
    # Step 1: Load selected features data
    X, y = load_selected_features_data()
    feature_names = list(X.columns)
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 3: Initialize models
    models = initialize_models()
    
    # Step 4: Train and evaluate models
    print(f"\n=== TRAINING AND EVALUATION ===")
    model_results = {}
    
    for model_name, model in models.items():
        results = train_and_evaluate_model(
            model, model_name, X_train, X_test, y_train, y_test
        )
        model_results[model_name] = results
    
    # Step 5: Cross-validation analysis
    cv_results = cross_validate_models(models, X, y)
    
    # Step 6: Create results summary
    results_df, best_f1, best_auc = create_results_summary(model_results)
    
    # Step 7: Analyze feature importance
    importance_results = analyze_feature_importance(model_results, feature_names)
    
    # Step 8: Save results
    save_model_results(model_results, cv_results, results_df)
    
    print(f"\n✅ TASK 2.4 COMPLETE!")
    print(f"All classification models trained and evaluated successfully.")
    print(f"Best performing model (F1-Score): {best_f1}")
    print(f"Best performing model (AUC): {best_auc}")
    
    return model_results, cv_results, results_df, importance_results

if __name__ == "__main__":
    model_results, cv_results, summary_df, feature_importance = main()