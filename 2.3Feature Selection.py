# Task 2.3: Feature Selection
# Comprehensive Feature Selection using Multiple Methods

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

def load_preprocessed_data():
    """
    Load the preprocessed and scaled data
    """
    print("=== LOADING PREPROCESSED DATA ===")
    
    # Load cleaned data
    heart_data = pd.read_csv('heart_disease_cleaned.csv')
    
    # Separate features and target
    X = heart_data.drop('target', axis=1)
    y = heart_data['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"Data loaded successfully!")
    print(f"Features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(X_scaled.columns)}")
    
    return X_scaled, y, scaler

def random_forest_feature_importance(X_scaled, y):
    """
    Method 1: Feature importance using Random Forest
    """
    print("\n=== METHOD 1: RANDOM FOREST FEATURE IMPORTANCE ===")
    
    # Train Random Forest
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_scaled, y)
    
    # Get feature importance scores
    feature_importance = pd.DataFrame({
        'feature': X_scaled.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature importance ranking:")
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Select top 8 features
    top_features_rf = feature_importance.head(8)['feature'].tolist()
    print(f"\nTop 8 features by Random Forest: {top_features_rf}")
    
    return feature_importance, top_features_rf

def recursive_feature_elimination(X_scaled, y, n_features=8):
    """
    Method 2: Recursive Feature Elimination with Logistic Regression
    """
    print(f"\n=== METHOD 2: RECURSIVE FEATURE ELIMINATION (RFE) ===")
    
    # Initialize base estimator
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Apply RFE
    rfe_selector = RFE(estimator=lr_model, n_features_to_select=n_features)
    X_rfe = rfe_selector.fit_transform(X_scaled, y)
    
    # Get selected features
    rfe_selected_features = X_scaled.columns[rfe_selector.support_].tolist()
    
    # Get feature rankings
    rfe_rankings = pd.DataFrame({
        'feature': X_scaled.columns,
        'selected': rfe_selector.support_,
        'ranking': rfe_selector.ranking_
    }).sort_values('ranking')
    
    print("RFE feature rankings:")
    for idx, row in rfe_rankings.iterrows():
        status = "SELECTED" if row['selected'] else f"Rank {row['ranking']}"
        print(f"{row['feature']}: {status}")
    
    print(f"\nRFE selected features: {rfe_selected_features}")
    
    return rfe_rankings, rfe_selected_features

def statistical_feature_selection(X_scaled, y, k=8):
    """
    Method 3: Statistical Feature Selection using ANOVA F-test
    """
    print(f"\n=== METHOD 3: STATISTICAL FEATURE SELECTION (ANOVA F-test) ===")
    
    # Apply SelectKBest with f_classif
    f_selector = SelectKBest(score_func=f_classif, k=k)
    X_f_test = f_selector.fit_transform(X_scaled, y)
    
    # Get feature scores
    f_test_scores = pd.DataFrame({
        'feature': X_scaled.columns,
        'f_score': f_selector.scores_,
        'p_value': f_selector.pvalues_,
        'selected': f_selector.get_support()
    }).sort_values('f_score', ascending=False)
    
    print("F-test feature scores:")
    for idx, row in f_test_scores.iterrows():
        status = "SELECTED" if row['selected'] else "Not selected"
        print(f"{row['feature']}: F-score={row['f_score']:.4f}, p-value={row['p_value']:.6f} ({status})")
    
    # Get selected features
    f_test_selected = f_test_scores[f_test_scores['selected']]['feature'].tolist()
    print(f"\nF-test selected features: {f_test_selected}")
    
    return f_test_scores, f_test_selected

def consensus_feature_selection(rf_features, rfe_features, f_test_features, X_scaled):
    """
    Combine results from all methods using voting
    """
    print(f"\n=== CONSENSUS FEATURE SELECTION ===")
    
    # Count votes for each feature
    all_features = set(rf_features) | set(rfe_features) | set(f_test_features)
    feature_votes = {}
    
    for feature in X_scaled.columns:
        votes = 0
        methods = []
        if feature in rf_features:
            votes += 1
            methods.append("RF")
        if feature in rfe_features:
            votes += 1
            methods.append("RFE")
        if feature in f_test_features:
            votes += 1
            methods.append("F-test")
        
        feature_votes[feature] = {
            'votes': votes,
            'methods': methods
        }
    
    # Display voting results
    print("Feature selection voting results:")
    for feature, info in sorted(feature_votes.items(), key=lambda x: x[1]['votes'], reverse=True):
        methods_str = ', '.join(info['methods']) if info['methods'] else "None"
        print(f"{feature}: {info['votes']} votes ({methods_str})")
    
    # Select features with at least 2 votes
    final_selected_features = [
        feature for feature, info in feature_votes.items() 
        if info['votes'] >= 2
    ]
    
    print(f"\nFinal selected features (≥2 votes): {final_selected_features}")
    print(f"Number of selected features: {len(final_selected_features)}")
    print(f"Feature reduction: {len(final_selected_features)}/{len(X_scaled.columns)} = {len(final_selected_features)/len(X_scaled.columns)*100:.1f}% retained")
    
    return final_selected_features, feature_votes

def create_selected_features_dataset(X_scaled, y, selected_features):
    """
    Create dataset with only selected features
    """
    print(f"\n=== CREATING SELECTED FEATURES DATASET ===")
    
    # Create dataset with selected features only
    X_selected = X_scaled[selected_features]
    
    # Combine with target
    selected_dataset = X_selected.copy()
    selected_dataset['target'] = y
    
    # Save to CSV
    selected_dataset.to_csv('heart_disease_selected_features.csv', index=False)
    
    print(f"Selected features dataset created and saved!")
    print(f"Selected dataset shape: {selected_dataset.shape}")
    print(f"Selected features: {list(X_selected.columns)}")
    
    return X_selected, selected_dataset

def save_feature_selection_results(feature_importance, rfe_rankings, f_test_scores, 
                                 final_features, feature_votes, X_scaled):
    """
    Save all feature selection results for analysis and visualization
    """
    print(f"\n=== SAVING FEATURE SELECTION RESULTS ===")
    
    # Prepare comprehensive results
    feature_selection_results = {
        'original_features': X_scaled.columns.tolist(),
        'random_forest_importance': {
            'features': feature_importance['feature'].tolist(),
            'importance_scores': feature_importance['importance'].tolist()
        },
        'rfe_results': {
            'features': rfe_rankings['feature'].tolist(),
            'rankings': rfe_rankings['ranking'].tolist(),
            'selected': rfe_rankings['selected'].tolist()
        },
        'f_test_results': {
            'features': f_test_scores['feature'].tolist(),
            'f_scores': f_test_scores['f_score'].tolist(),
            'p_values': f_test_scores['p_value'].tolist(),
            'selected': f_test_scores['selected'].tolist()
        },
        'consensus_results': {
            'final_selected_features': final_features,
            'feature_votes': {k: {'votes': v['votes'], 'methods': v['methods']} 
                            for k, v in feature_votes.items()}
        },
        'summary': {
            'original_feature_count': len(X_scaled.columns),
            'selected_feature_count': len(final_features),
            'reduction_percentage': len(final_features) / len(X_scaled.columns) * 100
        }
    }
    
    # Save to JSON
    with open('feature_selection_results.json', 'w') as f:
        json.dump(feature_selection_results, f, indent=2)
    
    print("Feature selection results saved to 'feature_selection_results.json'")
    
    return feature_selection_results

def evaluate_feature_selection_impact(X_original, X_selected, y):
    """
    Quick evaluation of feature selection impact using a simple model
    """
    print(f"\n=== EVALUATING FEATURE SELECTION IMPACT ===")
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    # Simple logistic regression for comparison
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Cross-validation scores
    scores_original = cross_val_score(lr_model, X_original, y, cv=5, scoring='f1')
    scores_selected = cross_val_score(lr_model, X_selected, y, cv=5, scoring='f1')
    
    print(f"Cross-validation F1 scores (5-fold):")
    print(f"Original features ({X_original.shape[1]} features): {scores_original.mean():.4f} ± {scores_original.std():.4f}")
    print(f"Selected features ({X_selected.shape[1]} features): {scores_selected.mean():.4f} ± {scores_selected.std():.4f}")
    
    improvement = scores_selected.mean() - scores_original.mean()
    print(f"Performance change: {improvement:+.4f}")
    
    return scores_original, scores_selected

def main():
    """
    Main feature selection pipeline
    """
    print("=== TASK 2.3: FEATURE SELECTION ===")
    
    # Step 1: Load preprocessed data
    X_scaled, y, scaler = load_preprocessed_data()
    
    # Step 2: Random Forest feature importance
    rf_importance, rf_features = random_forest_feature_importance(X_scaled, y)
    
    # Step 3: Recursive Feature Elimination
    rfe_rankings, rfe_features = recursive_feature_elimination(X_scaled, y)
    
    # Step 4: Statistical feature selection
    f_test_scores, f_test_features = statistical_feature_selection(X_scaled, y)
    
    # Step 5: Consensus feature selection
    final_features, feature_votes = consensus_feature_selection(
        rf_features, rfe_features, f_test_features, X_scaled
    )
    
    # Step 6: Create selected features dataset
    X_selected, selected_dataset = create_selected_features_dataset(X_scaled, y, final_features)
    
    # Step 7: Save comprehensive results
    results = save_feature_selection_results(
        rf_importance, rfe_rankings, f_test_scores, final_features, feature_votes, X_scaled
    )
    
    # Step 8: Evaluate impact
    cv_scores_orig, cv_scores_selected = evaluate_feature_selection_impact(X_scaled, X_selected, y)
    
    print(f"\n✅ TASK 2.3 COMPLETE!")
    print(f"Feature selection completed successfully.")
    print(f"Reduced from {len(X_scaled.columns)} to {len(final_features)} features")
    print(f"Selected features: {final_features}")
    
    return X_selected, final_features, results

if __name__ == "__main__":
    X_selected, selected_features, selection_results = main()