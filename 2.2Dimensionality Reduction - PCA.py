# Task 2.2: Principal Component Analysis (PCA)
# Comprehensive PCA Implementation for Heart Disease Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

def load_preprocessed_data():
    """
    Load the preprocessed data from Task 2.1
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
    
    return X_scaled, y, scaler

def apply_full_pca(X_scaled):
    """
    Apply PCA with all components to understand variance distribution
    """
    print("\n=== APPLYING FULL PCA ANALYSIS ===")
    
    # Apply PCA to all components
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    
    # Calculate explained variance ratios
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("Explained variance ratio for each component:")
    for i, var_ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    
    print(f"\nCumulative explained variance ratio:")
    for i, cum_var_ratio in enumerate(cumulative_variance_ratio):
        print(f"PC1-PC{i+1}: {cum_var_ratio:.4f} ({cum_var_ratio*100:.2f}%)")
    
    return pca_full, X_pca_full, explained_variance_ratio, cumulative_variance_ratio

def determine_optimal_components(cumulative_variance_ratio, threshold=0.95):
    """
    Determine optimal number of components for specified variance threshold
    """
    print(f"\n=== DETERMINING OPTIMAL COMPONENTS ===")
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance_ratio >= threshold) + 1
    variance_retained = cumulative_variance_ratio[n_components_95 - 1]
    
    print(f"Components needed to retain {threshold*100}% variance: {n_components_95}")
    print(f"Actual variance retained: {variance_retained:.4f} ({variance_retained*100:.2f}%)")
    
    return n_components_95, variance_retained

def apply_optimal_pca(X_scaled, n_components):
    """
    Apply PCA with optimal number of components
    """
    print(f"\n=== APPLYING OPTIMAL PCA ({n_components} components) ===")
    
    # Apply PCA with optimal components
    pca_optimal = PCA(n_components=n_components)
    X_pca_optimal = pca_optimal.fit_transform(X_scaled)
    
    variance_retained = pca_optimal.explained_variance_ratio_.sum()
    
    print(f"Original feature space: {X_scaled.shape}")
    print(f"Reduced feature space: {X_pca_optimal.shape}")
    print(f"Variance retained: {variance_retained:.4f} ({variance_retained*100:.2f}%)")
    
    return pca_optimal, X_pca_optimal

def save_pca_results(X_pca_optimal, y, n_components):
    """
    Save PCA-transformed dataset
    """
    print(f"\n=== SAVING PCA RESULTS ===")
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(X_pca_optimal, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['target'] = y.values
    
    # Save to CSV
    pca_df.to_csv('heart_disease_pca.csv', index=False)
    
    print(f"PCA-transformed dataset saved as 'heart_disease_pca.csv'")
    print(f"Dataset shape: {pca_df.shape}")
    
    return pca_df

def analyze_component_contributions(pca_model, feature_names):
    """
    Analyze which original features contribute most to each principal component
    """
    print(f"\n=== ANALYZING COMPONENT CONTRIBUTIONS ===")
    
    # Get the components (loadings)
    components_df = pd.DataFrame(
        pca_model.components_[:5],  # Show first 5 components
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(5)]
    )
    
    print("Top contributing features for each Principal Component:")
    for i in range(5):
        pc_name = f'PC{i+1}'
        contributions = components_df.loc[pc_name].abs().sort_values(ascending=False)
        print(f"\n{pc_name} (explains {pca_model.explained_variance_ratio_[i]*100:.2f}% variance):")
        for feature, contribution in contributions.head().items():
            print(f"  {feature}: {contribution:.4f}")
    
    return components_df

def create_pca_visualization_data(explained_variance_ratio, cumulative_variance_ratio):
    """
    Prepare data for PCA visualizations
    """
    print(f"\n=== PREPARING VISUALIZATION DATA ===")
    
    pca_variance_data = {
        'component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance_ratio': cumulative_variance_ratio.tolist()
    }
    
    # Save for visualization
    with open('pca_variance_data.json', 'w') as f:
        json.dump(pca_variance_data, f)
    
    print("PCA variance data saved for visualization")
    
    return pca_variance_data

def create_biplot_data(pca_model, X_pca, y, feature_names):
    """
    Create data for PCA biplot visualization
    """
    print(f"\n=== CREATING BIPLOT DATA ===")
    
    # Prepare biplot data (first 2 components)
    biplot_data = {
        'pc1': X_pca[:, 0].tolist(),
        'pc2': X_pca[:, 1].tolist(),
        'target': y.tolist(),
        'loadings': {
            'features': feature_names,
            'pc1_loadings': pca_model.components_[0].tolist(),
            'pc2_loadings': pca_model.components_[1].tolist()
        }
    }
    
    with open('pca_biplot_data.json', 'w') as f:
        json.dump(biplot_data, f)
    
    print("PCA biplot data saved")
    
    return biplot_data

def main():
    """
    Main PCA pipeline
    """
    print("=== TASK 2.2: PRINCIPAL COMPONENT ANALYSIS (PCA) ===")
    
    # Step 1: Load preprocessed data
    X_scaled, y, scaler = load_preprocessed_data()
    feature_names = X_scaled.columns.tolist()
    
    # Step 2: Apply full PCA analysis
    pca_full, X_pca_full, explained_variance, cumulative_variance = apply_full_pca(X_scaled)
    
    # Step 3: Determine optimal number of components
    n_components_95, variance_retained = determine_optimal_components(cumulative_variance)
    
    # Step 4: Apply PCA with optimal components
    pca_optimal, X_pca_optimal = apply_optimal_pca(X_scaled, n_components_95)
    
    # Step 5: Save PCA results
    pca_df = save_pca_results(X_pca_optimal, y, n_components_95)
    
    # Step 6: Analyze component contributions
    components_df = analyze_component_contributions(pca_optimal, feature_names)
    
    # Step 7: Create visualization data
    variance_data = create_pca_visualization_data(explained_variance, cumulative_variance)
    biplot_data = create_biplot_data(pca_optimal, X_pca_optimal, y, feature_names)
    
    print(f"\n✅ TASK 2.2 COMPLETE!")
    print(f"PCA analysis completed successfully.")
    print(f"Dimensionality reduced from {X_scaled.shape[1]} to {n_components_95} features")
    print(f"Variance retained: {variance_retained:.4f} ({variance_retained*100:.2f}%)")
    
    return pca_optimal, X_pca_optimal, pca_df, components_df

if __name__ == "__main__":
    pca_model, X_pca, pca_df, components = main()