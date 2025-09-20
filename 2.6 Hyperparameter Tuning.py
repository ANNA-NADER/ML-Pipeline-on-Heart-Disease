# Task 2.6: Hyperparameter Tuning
# Comprehensive Hyperparameter Optimization for All Models

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from scipy.stats import uniform, randint

def load_data_and_models():
    """
    Load data and initialize baseline models
    """
    print("=== LOADING DATA AND BASELINE MODELS ===")
    
    # Load dataset with selected features
    selected_data = pd.read_csv('heart_disease_selected_features.csv')
    X = selected_data.drop('target', axis=1)
    y = selected_data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Initialize baseline models
    baseline_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    return X_train, X_test, y_train, y_test, baseline_models

def get_baseline_performance(models, X_train, X_test, y_train, y_test):
    """
    Get baseline performance for all models
    """
    print(f"\n=== BASELINE MODEL PERFORMANCE ===")
    
    baseline_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating baseline {model_name}...")
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        baseline_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
    
    return baseline_results

def define_hyperparameter_grids():
    """
    Define hyperparameter grids for each model
    """
    print(f"\n=== DEFINING HYPERPARAMETER GRIDS ===")
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Only used when penalty='elasticnet'
        },
        
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2', None]
        },
        
        'Random Forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        },
        
        'SVM': {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],  # Only used when kernel='poly'
            'coef0': [0.0, 0.1, 0.5, 1.0]  # Only used for poly/sigmoid
        }
    }
    
    for model_name, params in param_grids.items():
        print(f"\n{model_name} parameters:")
        for param, values in params.items():
            print(f"  {param}: {values}")
    
    return param_grids

def tune_logistic_regression(X_train, y_train, param_grid):
    """
    Tune Logistic Regression with custom parameter handling
    """
    print(f"\n--- Tuning Logistic Regression ---")
    
    # Custom parameter grid to handle elasticnet constraints
    custom_params = []
    
    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            if penalty == 'elasticnet':
                for solver in ['saga']:  # Only saga supports elasticnet
                    for l1_ratio in param_grid['l1_ratio']:
                        custom_params.append({
                            'C': C,
                            'penalty': penalty,
                            'solver': solver,
                            'l1_ratio': l1_ratio
                        })
            else:
                for solver in param_grid['solver']:
                    if (penalty == 'l1' and solver in ['liblinear', 'saga']) or \
                       (penalty == 'l2' and solver in ['liblinear', 'saga']):
                        custom_params.append({
                            'C': C,
                            'penalty': penalty,
                            'solver': solver
                        })
    
    # Random search over custom parameters
    model = LogisticRegression(random_state=42, max_iter=1000)
    search = GridSearchCV(
        model, custom_params[:50],  # Limit to avoid too many combinations
        cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print(f"  Best parameters: {search.best_params_}")
    print(f"  Best CV F1-score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search.best_score_

def tune_model_with_search(model, model_name, param_grid, X_train, y_train, search_type='random'):
    """
    Tune a model using GridSearchCV or RandomizedSearchCV
    """
    print(f"\n--- Tuning {model_name} ---")
    
    if search_type == 'random':
        # Use RandomizedSearchCV for larger parameter spaces
        search = RandomizedSearchCV(
            model, 
            param_grid,
            n_iter=100,  # Number of parameter settings to try
            cv=5,
            scoring='f1',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    else:
        # Use GridSearchCV for smaller parameter spaces
        search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
    
    search.fit(X_train, y_train)
    
    print(f"  Best parameters: {search.best_params_}")
    print(f"  Best CV F1-score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search.best_score_

def tune_all_models(baseline_models, param_grids, X_train, y_train):
    """
    Tune hyperparameters for all models
    """
    print(f"\n=== HYPERPARAMETER TUNING ===")
    print("This process may take several minutes...")
    
    tuned_models = {}
    tuning_results = {}
    
    for model_name, model in baseline_models.items():
        print(f"\n{'='*50}")
        print(f"TUNING {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            if model_name == 'Logistic Regression':
                # Special handling for Logistic Regression
                best_model, best_params, best_score = tune_logistic_regression(
                    X_train, y_train, param_grids[model_name]
                )
            elif model_name in ['Random Forest', 'SVM']:
                # Use RandomizedSearchCV for complex models
                best_model, best_params, best_score = tune_model_with_search(
                    model, model_name, param_grids[model_name], 
                    X_train, y_train, search_type='random'
                )
            else:
                # Use GridSearchCV for simpler models
                best_model, best_params, best_score = tune_model_with_search(
                    model, model_name, param_grids[model_name], 
                    X_train, y_train, search_type='grid'
                )
            
            tuned_models[model_name] = best_model
            tuning_results[model_name] = {
                'best_params': best_params,
                'best_cv_score': best_score
            }
            
        except Exception as e:
            print(f"Error tuning {model_name}: {e}")
            # Use baseline model if tuning fails
            tuned_models[model_name] = model
            tuning_results[model_name] = {
                'best_params': 'Tuning failed - using baseline',
                'best_cv_score': 0.0
            }
    
    return tuned_models, tuning_results

def evaluate_tuned_models(tuned_models, X_test, y_test):
    """
    Evaluate tuned models on test set
    """
    print(f"\n=== EVALUATING TUNED MODELS ON TEST SET ===")
    
    tuned_performance = {}
    
    for model_name, model in tuned_models.items():
        print(f"\nEvaluating tuned {model_name}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            tuned_performance[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            tuned_performance[model_name] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'auc_score': 0.0
            }
    
    return tuned_performance

def compare_baseline_vs_tuned(baseline_results, tuned_results):
    """
    Compare baseline vs tuned model performance
    """
    print(f"\n=== BASELINE vs TUNED PERFORMANCE COMPARISON ===")
    
    comparison_data = []
    
    for model_name in baseline_results.keys():
        baseline_f1 = baseline_results[model_name]['f1_score']
        tuned_f1 = tuned_results[model_name]['f1_score']
        improvement = tuned_f1 - baseline_f1
        
        baseline_auc = baseline_results[model_name]['auc_score']
        tuned_auc = tuned_results[model_name]['auc_score']
        auc_improvement = tuned_auc - baseline_auc
        
        comparison_data.append({
            'Model': model_name,
            'Baseline F1': baseline_f1,
            'Tuned F1': tuned_f1,
            'F1 Improvement': improvement,
            'Baseline AUC': baseline_auc,
            'Tuned AUC': tuned_auc,
            'AUC Improvement': auc_improvement
        })
    
    comparison_df = pd.DataFrame(comparison_data).round(4)
    print(comparison_df)
    
    # Find best performing models
    best_tuned_f1 = comparison_df.loc[comparison_df['Tuned F1'].idxmax(), 'Model']
    best_tuned_auc = comparison_df.loc[comparison_df['Tuned AUC'].idxmax(), 'Model']
    
    print(f"\nBest tuned model (F1-Score): {best_tuned_f1}")
    print(f"Best tuned model (AUC): {best_tuned_auc}")
    
    return comparison_df, best_tuned_f1, best_tuned_auc

def save_tuning_results(tuning_results, comparison_df, best_tuned_models):
    """
    Save all hyperparameter tuning results
    """
    print(f"\n=== SAVING TUNING RESULTS ===")
    
    # Save comparison results
    comparison_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print("Performance comparison saved to 'hyperparameter_tuning_results.csv'")
    
    # Save detailed tuning results
    detailed_results = {}
    for model_name, results in tuning_results.items():
        detailed_results[model_name] = {
            'best_parameters': str(results['best_params']),
            'best_cv_score': results['best_cv_score']
        }
    
    with open('detailed_tuning_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print("Detailed tuning results saved to 'detailed_tuning_results.json'")
    
    # Save best models (model objects)
    import joblib
    for model_name, model in best_tuned_models.items():
        filename = f"best_{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        print(f"Best {model_name} saved to '{filename}'")

def main():
    """
    Main hyperparameter tuning pipeline
    """
    print("=== TASK 2.6: HYPERPARAMETER TUNING ===")
    
    # Step 1: Load data and baseline models
    X_train, X_test, y_train, y_test, baseline_models = load_data_and_models()
    
    # Step 2: Get baseline performance
    baseline_results = get_baseline_performance(baseline_models, X_train, X_test, y_train, y_test)
    
    # Step 3: Define hyperparameter grids
    param_grids = define_hyperparameter_grids()
    
    # Step 4: Tune all models
    tuned_models, tuning_results = tune_all_models(baseline_models, param_grids, X_train, y_train)
    
    # Step 5: Evaluate tuned models
    tuned_performance = evaluate_tuned_models(tuned_models, X_test, y_test)
    
    # Step 6: Compare baseline vs tuned
    comparison_df, best_f1_model, best_auc_model = compare_baseline_vs_tuned(
        baseline_results, tuned_performance
    )
    
    # Step 7: Save results
    save_tuning_results(tuning_results, comparison_df, tuned_models)
    
    print(f"\n✅ TASK 2.6 COMPLETE!")
    print(f"Hyperparameter tuning completed successfully.")
    print(f"Best F1 model: {best_f1_model}")
    print(f"Best AUC model: {best_auc_model}")
    
    return tuned_models, tuning_results, comparison_df

if __name__ == "__main__":
    tuned_models, tuning_results, comparison = main()