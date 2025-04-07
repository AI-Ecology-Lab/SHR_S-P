#!/usr/bin/env python3
"""
Joint Species Distribution Models (JSDMs)
-----------------------------------------
Purpose: Model the occurrence or abundance of multiple species simultaneously, 
accounting for both shared environmental responses and species interactions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
from scipy import stats

# Create output directories if they don't exist
os.makedirs('../analysis_results', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('../SHR_2022-2023_fauna_ctd_calcO2_pressure.csv')

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day

# Select species columns
species_columns = ['Sablefish', 'Sea Stars', 'Bubble', 'Crabs', 'Hagfish', 
                  'Euphausia', 'Liponema', 'Flatfish', 'Rockfish', 'Eelpout']

# Select environmental variables
env_columns = ['Temperature', 'Conductivity', 'Pressure', 'Salinity', 'Oxygen (ml/l)', 'PressurePSI']

# Fill missing values if any
data = data.fillna(method='ffill')

# Check if there's enough variation in species data
species_var = data[species_columns].var()
valid_species = species_var[species_var > 0].index.tolist()

if len(valid_species) < 2:
    print("Not enough variation in species data for JSDM analysis")
    with open('../analysis_results/jsdm_results.txt', 'w') as f:
        f.write("JSDM analysis could not be performed: Not enough variation in species data\n")
    exit()

print(f"Using {len(valid_species)} species with variation: {valid_species}")

# Convert to presence/absence data for simplicity
species_pa = data[valid_species].copy()
for col in valid_species:
    species_pa[col] = (species_pa[col] > 0).astype(int)

# Standardize environmental variables
env_data = data[env_columns].copy()
env_scaled = StandardScaler().fit_transform(env_data)
env_scaled_df = pd.DataFrame(env_scaled, columns=env_columns)

# 1. Calculate species co-occurrence patterns
cooccurrence_matrix = np.zeros((len(valid_species), len(valid_species)))
p_values_matrix = np.zeros((len(valid_species), len(valid_species)))

for i, sp1 in enumerate(valid_species):
    for j, sp2 in enumerate(valid_species):
        if i != j:
            # Calculate phi coefficient (correlation for binary data)
            contingency_table = pd.crosstab(species_pa[sp1], species_pa[sp2])
            
            # Handle cases where some values might be missing in the contingency table
            if contingency_table.shape == (2, 2):
                # Calculate chi-square test
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Calculate phi coefficient
                n = contingency_table.sum().sum()
                phi = np.sqrt(chi2 / n)
                
                # Determine sign of correlation
                if contingency_table.iloc[0, 0] * contingency_table.iloc[1, 1] < contingency_table.iloc[0, 1] * contingency_table.iloc[1, 0]:
                    phi = -phi
                
                cooccurrence_matrix[i, j] = phi
                p_values_matrix[i, j] = p
            else:
                cooccurrence_matrix[i, j] = 0
                p_values_matrix[i, j] = 1

# 2. Identify environmental drivers for each species
env_correlations = {}
for species in valid_species:
    correlations = {}
    for env_var in env_columns:
        corr, p_value = stats.pointbiserialr(species_pa[species], data[env_var])
        correlations[env_var] = {'correlation': corr, 'p_value': p_value}
    env_correlations[species] = correlations

# 3. Cluster species based on environmental responses
# Create a matrix of species-environment correlations
env_corr_matrix = np.zeros((len(valid_species), len(env_columns)))
for i, species in enumerate(valid_species):
    for j, env_var in enumerate(env_columns):
        env_corr_matrix[i, j] = env_correlations[species][env_var]['correlation']

# Apply K-means clustering to group species by environmental response
n_clusters = min(3, len(valid_species))  # Adjust based on number of valid species
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(env_corr_matrix)

# Create cluster assignments
cluster_assignments = {species: cluster for species, cluster in zip(valid_species, clusters)}

# Save results
with open('../analysis_results/jsdm_results.txt', 'w') as f:
    f.write("Joint Species Distribution Model Results\n")
    f.write("="*50 + "\n\n")
    
    f.write("1. Species Co-occurrence Patterns\n")
    f.write("-"*30 + "\n")
    f.write("Phi coefficient matrix (correlation for binary data):\n\n")
    
    # Write header
    f.write("           " + " ".join([f"{sp[:7]:8s}" for sp in valid_species]) + "\n")
    
    # Write matrix
    for i, sp1 in enumerate(valid_species):
        f.write(f"{sp1[:10]:10s} " + " ".join([f"{cooccurrence_matrix[i, j]:8.4f}" for j in range(len(valid_species))]) + "\n")
    
    f.write("\nSignificant co-occurrences (p < 0.05):\n")
    for i, sp1 in enumerate(valid_species):
        for j, sp2 in enumerate(valid_species):
            if i != j and p_values_matrix[i, j] < 0.05:
                relation = "positive" if cooccurrence_matrix[i, j] > 0 else "negative"
                f.write(f"  - {sp1} and {sp2}: {relation} association (phi = {cooccurrence_matrix[i, j]:.4f}, p = {p_values_matrix[i, j]:.4f})\n")
    
    f.write("\n\n2. Environmental Drivers for Each Species\n")
    f.write("-"*40 + "\n")
    
    for species in valid_species:
        f.write(f"\n{species}:\n")
        # Sort by absolute correlation strength
        sorted_correlations = sorted(env_correlations[species].items(), 
                                    key=lambda x: abs(x[1]['correlation']), 
                                    reverse=True)
        
        for env_var, stats in sorted_correlations:
            corr = stats['correlation']
            p_val = stats['p_value']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            f.write(f"  - {env_var}: r = {corr:.4f} (p = {p_val:.4f}) {significance}\n")
    
    f.write("\n\n3. Species Clusters Based on Environmental Responses\n")
    f.write("-"*50 + "\n")
    
    for cluster_id in range(n_clusters):
        cluster_species = [sp for sp, cl in cluster_assignments.items() if cl == cluster_id]
        f.write(f"\nCluster {cluster_id+1}:\n")
        for species in cluster_species:
            f.write(f"  - {species}\n")
        
        # Calculate average environmental response for this cluster
        f.write("\n  Average environmental response:\n")
        cluster_indices = [i for i, sp in enumerate(valid_species) if sp in cluster_species]
        cluster_env_response = env_corr_matrix[cluster_indices].mean(axis=0)
        
        for j, env_var in enumerate(env_columns):
            f.write(f"    {env_var}: {cluster_env_response[j]:.4f}\n")

# Create visualizations

# 1. Co-occurrence heatmap
plt.figure(figsize=(10, 8))
mask = np.eye(len(valid_species), dtype=bool)  # Mask the diagonal
sns.heatmap(cooccurrence_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
           xticklabels=valid_species, yticklabels=valid_species, mask=mask,
           vmin=-1, vmax=1)
plt.title('Species Co-occurrence Patterns (Phi Coefficient)')
plt.tight_layout()
plt.savefig('../visualizations/jsdm_cooccurrence_heatmap.png')
plt.close()

# 2. Environmental drivers heatmap
env_corr_df = pd.DataFrame(env_corr_matrix, index=valid_species, columns=env_columns)
plt.figure(figsize=(12, 8))
sns.heatmap(env_corr_df, annot=True, cmap='coolwarm', fmt='.2f', 
           vmin=-1, vmax=1)
plt.title('Species-Environment Correlations')
plt.tight_layout()
plt.savefig('../visualizations/jsdm_environment_correlations.png')
plt.close()

# 3. Cluster visualization using PCA
if len(valid_species) > 2:
    pca = PCA(n_components=2)
    env_corr_pca = pca.fit_transform(env_corr_matrix)
    
    plt.figure(figsize=(10, 8))
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, sp in enumerate(valid_species) if cluster_assignments[sp] == cluster_id]
        plt.scatter(env_corr_pca[cluster_indices, 0], env_corr_pca[cluster_indices, 1], 
                   label=f'Cluster {cluster_id+1}', s=100)
        
        # Add species labels
        for idx in cluster_indices:
            plt.annotate(valid_species[idx], (env_corr_pca[idx, 0], env_corr_pca[idx, 1]),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.title('Species Clusters Based on Environmental Responses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/jsdm_species_clusters.png')
    plt.close()

print("\nJSDM analysis completed. Results saved to analysis_results/jsdm_results.txt")
print("Visualizations saved to visualizations/ directory")
