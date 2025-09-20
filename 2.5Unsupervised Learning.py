# Task 2.5: Unsupervised Learning - Clustering
# Comprehensive Clustering Analysis using K-Means and Hierarchical Clustering

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import json

def load_selected_features_data():
    """
    Load the dataset with selected features for clustering
    """
    print("=== LOADING DATA FOR CLUSTERING ===")
    
    # Load dataset with selected features
    selected_data = pd.read_csv('heart_disease_selected_features.csv')
    
    # Separate features and target
    X = selected_data.drop('target', axis=1)
    y = selected_data['target']  # We'll use this for evaluation
    
    print(f"Data loaded successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution (for evaluation): {y.value_counts().to_dict()}")
    
    return X, y

def determine_optimal_k_means(X, k_range=range(2, 11)):
    """
    Use elbow method and silhouette analysis to determine optimal K for K-means
    """
    print(f"\n=== DETERMINING OPTIMAL K FOR K-MEANS ===")
    print(f"Testing K values: {list(k_range)}")
    
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"  Testing K={k}...")
        
        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X, cluster_labels)
        
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        
        print(f"    Inertia: {inertia:.2f}, Silhouette Score: {sil_score:.4f}")
    
    # Find optimal K based on silhouette score
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = list(k_range)[optimal_k_idx]
    best_silhouette = silhouette_scores[optimal_k_idx]
    
    print(f"\nOptimal K: {optimal_k}")
    print(f"Best Silhouette Score: {best_silhouette:.4f}")
    
    # Save elbow data for visualization
    elbow_data = {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k
    }
    
    with open('elbow_curve_data.json', 'w') as f:
        json.dump(elbow_data, f)
    
    return optimal_k, elbow_data

def apply_kmeans_clustering(X, y, optimal_k):
    """
    Apply K-means clustering with optimal K
    """
    print(f"\n=== APPLYING K-MEANS CLUSTERING (K={optimal_k}) ===")
    
    # Apply K-means with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Evaluate clustering performance
    ari = adjusted_rand_score(y, kmeans_labels)
    ami = adjusted_mutual_info_score(y, kmeans_labels)
    silhouette = silhouette_score(X, kmeans_labels)
    
    print(f"K-Means Clustering Results:")
    print(f"  Number of clusters: {optimal_k}")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Adjusted Mutual Info: {ami:.4f}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    # Analyze cluster composition vs actual labels
    print(f"\nCluster composition analysis:")
    cluster_composition = pd.crosstab(
        kmeans_labels, y, 
        margins=True, margins_name='Total'
    )
    print(cluster_composition)
    
    # Cluster centers
    print(f"\nCluster centers:")
    centers_df = pd.DataFrame(kmeans.cluster_centers_, 
                            columns=X.columns,
                            index=[f'Cluster {i}' for i in range(optimal_k)])
    print(centers_df.round(4))
    
    return {
        'model': kmeans,
        'labels': kmeans_labels,
        'ari': ari,
        'ami': ami,
        'silhouette': silhouette,
        'inertia': kmeans.inertia_,
        'cluster_centers': centers_df,
        'composition': cluster_composition
    }

def apply_hierarchical_clustering(X, y, n_clusters):
    """
    Apply Hierarchical (Agglomerative) clustering
    """
    print(f"\n=== APPLYING HIERARCHICAL CLUSTERING (n_clusters={n_clusters}) ===")
    
    # Apply different linkage methods
    linkage_methods = ['ward', 'complete', 'average', 'single']
    hierarchical_results = {}
    
    for linkage_method in linkage_methods:
        print(f"\n  Testing linkage: {linkage_method}")
        
        try:
            # Apply hierarchical clustering
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=linkage_method
            )
            hierarchical_labels = hierarchical.fit_predict(X)
            
            # Evaluate clustering performance
            ari = adjusted_rand_score(y, hierarchical_labels)
            ami = adjusted_mutual_info_score(y, hierarchical_labels)
            silhouette = silhouette_score(X, hierarchical_labels)
            
            print(f"    Adjusted Rand Index: {ari:.4f}")
            print(f"    Adjusted Mutual Info: {ami:.4f}")
            print(f"    Silhouette Score: {silhouette:.4f}")
            
            # Cluster composition
            composition = pd.crosstab(hierarchical_labels, y, margins=True, margins_name='Total')
            
            hierarchical_results[linkage_method] = {
                'model': hierarchical,
                'labels': hierarchical_labels,
                'ari': ari,
                'ami': ami,
                'silhouette': silhouette,
                'composition': composition
            }
            
        except Exception as e:
            print(f"    Error with {linkage_method}: {e}")
            continue
    
    # Find best linkage method
    if hierarchical_results:
        best_linkage = max(hierarchical_results.keys(), 
                          key=lambda k: hierarchical_results[k]['silhouette'])
        
        print(f"\nBest linkage method: {best_linkage}")
        print(f"Best results:")
        best_result = hierarchical_results[best_linkage]
        print(f"  Adjusted Rand Index: {best_result['ari']:.4f}")
        print(f"  Adjusted Mutual Info: {best_result['ami']:.4f}")
        print(f"  Silhouette Score: {best_result['silhouette']:.4f}")
        
        print(f"\nBest hierarchical cluster composition:")
        print(best_result['composition'])
        
        return best_result, hierarchical_results
    else:
        print("No hierarchical clustering results obtained")
        return None, {}

def create_dendrogram_data(X, max_samples=100):
    """
    Create dendrogram for hierarchical clustering visualization
    Note: Limited to max_samples for visualization purposes
    """
    print(f"\n=== CREATING DENDROGRAM DATA ===")
    
    if len(X) > max_samples:
        print(f"Sampling {max_samples} points for dendrogram (from {len(X)} total)")
        sample_indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        X_sample = X
    
    # Calculate linkage matrix
    linkage_matrix = linkage(X_sample, method='ward')
    
    # Save dendrogram data
    dendrogram_data = {
        'linkage_matrix': linkage_matrix.tolist(),
        'sample_size': len(X_sample),
        'total_size': len(X)
    }
    
    with open('dendrogram_data.json', 'w') as f:
        json.dump(dendrogram_data, f)
    
    print(f"Dendrogram data saved for {len(X_sample)} samples")
    
    return dendrogram_data

def compare_clustering_methods(kmeans_results, hierarchical_results):
    """
    Compare K-means and Hierarchical clustering results
    """
    print(f"\n=== CLUSTERING METHODS COMPARISON ===")
    
    if hierarchical_results:
        comparison_data = {
            'K-Means': {
                'ARI': kmeans_results['ari'],
                'AMI': kmeans_results['ami'],
                'Silhouette Score': kmeans_results['silhouette']
            },
            'Hierarchical': {
                'ARI': hierarchical_results['ari'],
                'AMI': hierarchical_results['ami'],
                'Silhouette Score': hierarchical_results['silhouette']
            }
        }
        
        comparison_df = pd.DataFrame(comparison_data).round(4)
        print(comparison_df)
        
        # Determine better method
        if kmeans_results['silhouette'] > hierarchical_results['silhouette']:
            better_method = 'K-Means'
            better_score = kmeans_results['silhouette']
        else:
            better_method = 'Hierarchical'
            better_score = hierarchical_results['silhouette']
        
        print(f"\nBetter performing method: {better_method}")
        print(f"Higher Silhouette Score: {better_score:.4f}")
        
        return comparison_df, better_method
    
    else:
        print("Cannot compare - hierarchical clustering failed")
        return None, 'K-Means'

def analyze_cluster_characteristics(X, kmeans_results):
    """
    Analyze characteristics of each cluster
    """
    print(f"\n=== CLUSTER CHARACTERISTICS ANALYSIS ===")
    
    # Add cluster labels to original data
    X_with_clusters = X.copy()
    X_with_clusters['cluster'] = kmeans_results['labels']
    
    # Calculate cluster statistics
    cluster_stats = X_with_clusters.groupby('cluster').agg(['mean', 'std']).round(4)
    
    print("Cluster Statistics (mean ± std):")
    for cluster_id in sorted(X_with_clusters['cluster'].unique()):
        print(f"\nCluster {cluster_id}:")
        cluster_data = X_with_clusters[X_with_clusters['cluster'] == cluster_id]
        print(f"  Size: {len(cluster_data)} samples")
        
        for feature in X.columns:
            mean_val = cluster_data[feature].mean()
            std_val = cluster_data[feature].std()
            print(f"  {feature}: {mean_val:.4f} ± {std_val:.4f}")
    
    return cluster_stats

def save_clustering_results(X, y, kmeans_results, hierarchical_results, comparison_df):
    """
    Save all clustering results
    """
    print(f"\n=== SAVING CLUSTERING RESULTS ===")
    
    # Create comprehensive results dataset
    results_data = X.copy()
    results_data['actual_label'] = y
    results_data['kmeans_cluster'] = kmeans_results['labels']
    
    if hierarchical_results:
        results_data['hierarchical_cluster'] = hierarchical_results['labels']
    
    # Save to CSV
    results_data.to_csv('clustering_results.csv', index=False)
    print("Clustering results saved to 'clustering_results.csv'")
    
    # Save summary results
    if comparison_df is not None:
        comparison_df.to_csv('clustering_comparison.csv')
        print("Clustering comparison saved to 'clustering_comparison.csv'")
    
    # Save detailed results
    detailed_results = {
        'kmeans': {
            'optimal_k': len(np.unique(kmeans_results['labels'])),
            'ari': kmeans_results['ari'],
            'ami': kmeans_results['ami'],
            'silhouette_score': kmeans_results['silhouette'],
            'inertia': kmeans_results['inertia'],
            'cluster_centers': kmeans_results['cluster_centers'].to_dict()
        }
    }
    
    if hierarchical_results:
        detailed_results['hierarchical'] = {
            'n_clusters': len(np.unique(hierarchical_results['labels'])),
            'ari': hierarchical_results['ari'],
            'ami': hierarchical_results['ami'],
            'silhouette_score': hierarchical_results['silhouette']
        }
    
    with open('detailed_clustering_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("Detailed clustering results saved")

def main():
    """
    Main unsupervised learning pipeline
    """
    print("=== TASK 2.5: UNSUPERVISED LEARNING - CLUSTERING ===")
    
    # Step 1: Load data
    X, y = load_selected_features_data()
    
    # Step 2: Determine optimal K for K-means
    optimal_k, elbow_data = determine_optimal_k_means(X)
    
    # Step 3: Apply K-means clustering
    kmeans_results = apply_kmeans_clustering(X, y, optimal_k)
    
    # Step 4: Apply hierarchical clustering
    hierarchical_best, hierarchical_all = apply_hierarchical_clustering(X, y, optimal_k)
    
    # Step 5: Create dendrogram data
    dendrogram_data = create_dendrogram_data(X)
    
    # Step 6: Compare clustering methods
    comparison_df, better_method = compare_clustering_methods(kmeans_results, hierarchical_best)
    
    # Step 7: Analyze cluster characteristics
    cluster_stats = analyze_cluster_characteristics(X, kmeans_results)
    
    # Step 8: Save all results
    save_clustering_results(X, y, kmeans_results, hierarchical_best, comparison_df)
    
    print(f"\n✅ TASK 2.5 COMPLETE!")
    print(f"Clustering analysis completed successfully.")
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Best performing method: {better_method}")
    if hierarchical_best:
        print(f"Best silhouette score: {max(kmeans_results['silhouette'], hierarchical_best['silhouette']):.4f}")
    else:
        print(f"K-means silhouette score: {kmeans_results['silhouette']:.4f}")
    
    return kmeans_results, hierarchical_best, comparison_df, cluster_stats

if __name__ == "__main__":
    kmeans_result, hierarchical_result, comparison, stats = main()