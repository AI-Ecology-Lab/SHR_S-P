#!/usr/bin/env python3
"""
Canonical Correspondence Analysis (CCA)
---------------------------------------
Purpose: Explore how community composition relates directly to environmental gradients.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# Create output directories if they don't exist
os.makedirs('../analysis_results', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('../upload/SHR_2022-2023_fauna_ctd_calcO2_pressure.csv')

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

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
    print("Not enough variation in species data for CCA analysis")
    with open('../analysis_results/cca_results.txt', 'w') as f:
        f.write("CCA analysis could not be performed: Not enough variation in species data\n")
    exit()

print(f"Using {len(valid_species)} species with variation: {valid_species}")

# Extract species and environmental data
Y = data[valid_species].values  # Species data
X = data[env_columns].values    # Environmental data

# Standardize environmental data
X_std = StandardScaler().fit_transform(X)

# Since we don't have a direct CCA implementation in scikit-learn, we'll implement a simplified version
# using PCA and regression to approximate CCA

# Step 1: Run PCA on species data
pca_species = PCA()
species_scores = pca_species.fit_transform(Y)

# Step 2: Run PCA on environmental data
pca_env = PCA()
env_scores = pca_env.fit_transform(X_std)

# Step 3: Calculate correlations between principal components
correlations = np.zeros((pca_species.n_components_, pca_env.n_components_))
for i in range(min(pca_species.n_components_, pca_env.n_components_)):
    for j in range(min(pca_species.n_components_, pca_env.n_components_)):
        correlations[i, j] = np.corrcoef(species_scores[:, i], env_scores[:, j])[0, 1]

# Save correlation results
with open('../analysis_results/cca_results.txt', 'w') as f:
    f.write("Canonical Correspondence Analysis Results\n")
    f.write("="*50 + "\n\n")
    
    f.write("Correlations between species and environmental principal components:\n")
    f.write("-"*70 + "\n")
    f.write("           " + " ".join([f"Env PC{i+1:2d}".ljust(10) for i in range(min(5, pca_env.n_components_))]) + "\n")
    
    for i in range(min(5, pca_species.n_components_)):
        f.write(f"Species PC{i+1:2d} " + " ".join([f"{correlations[i, j]:.4f}".ljust(10) for j in range(min(5, pca_env.n_components_))]) + "\n")
    
    f.write("\n\nSpecies PCA explained variance:\n")
    f.write("-"*30 + "\n")
    for i, var in enumerate(pca_species.explained_variance_ratio_[:5]):
        f.write(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)\n")
    
    f.write("\n\nEnvironmental PCA explained variance:\n")
    f.write("-"*30 + "\n")
    for i, var in enumerate(pca_env.explained_variance_ratio_[:5]):
        f.write(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)\n")
    
    # Calculate and save species-environment correlations
    f.write("\n\nSpecies-Environment Correlations:\n")
    f.write("-"*50 + "\n")
    
    for i, species in enumerate(valid_species):
        f.write(f"\n{species}:\n")
        for j, env_var in enumerate(env_columns):
            corr, p_value = stats.pearsonr(data[species], data[env_var])
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            f.write(f"  - {env_var}: r = {corr:.4f} (p = {p_value:.4f}) {significance}\n")

# Create CCA biplot
plt.figure(figsize=(12, 10))

# Plot species scores
for i, species in enumerate(valid_species):
    plt.scatter(species_scores[:, 0][data[species] > 0], 
                species_scores[:, 1][data[species] > 0], 
                alpha=0.5, s=data[species][data[species] > 0]*10, 
                label=species)

# Add species labels
for i, species in enumerate(valid_species):
    # Calculate centroid of points for each species
    mask = data[species] > 0
    if mask.sum() > 0:  # Only if there are points
        x_centroid = species_scores[:, 0][mask].mean()
        y_centroid = species_scores[:, 1][mask].mean()
        plt.text(x_centroid, y_centroid, species, fontsize=12, 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

# Plot environmental variable vectors
scaling_factor = 5  # Adjust this to make arrows visible
for i, env_var in enumerate(env_columns):
    # Calculate correlation with the first two PCs
    corr_x = np.corrcoef(data[env_var], species_scores[:, 0])[0, 1]
    corr_y = np.corrcoef(data[env_var], species_scores[:, 1])[0, 1]
    
    # Plot arrow
    plt.arrow(0, 0, corr_x * scaling_factor, corr_y * scaling_factor, 
              head_width=0.1, head_length=0.2, fc='red', ec='red')
    
    # Add label
    plt.text(corr_x * scaling_factor * 1.1, corr_y * scaling_factor * 1.1, 
             env_var, color='red', ha='center', va='center')

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.title('CCA Biplot: Species and Environmental Variables')
plt.xlabel(f'PC1 ({pca_species.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca_species.explained_variance_ratio_[1]*100:.2f}%)')

# Create custom legend for environmental variables
env_patch = mpatches.Patch(color='red', label='Environmental Variables')
plt.legend(handles=[env_patch], loc='upper right')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/cca_biplot.png')
plt.close()

# Create heatmap of species-environment correlations
corr_matrix = np.zeros((len(valid_species), len(env_columns)))
p_values = np.zeros((len(valid_species), len(env_columns)))

for i, species in enumerate(valid_species):
    for j, env_var in enumerate(env_columns):
        corr, p_value = stats.pearsonr(data[species], data[env_var])
        corr_matrix[i, j] = corr
        p_values[i, j] = p_value

plt.figure(figsize=(12, 8))
im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation coefficient')

# Add significance markers
for i in range(len(valid_species)):
    for j in range(len(env_columns)):
        if p_values[i, j] < 0.05:
            marker = '*' if p_values[i, j] < 0.05 else '**' if p_values[i, j] < 0.01 else '***'
            plt.text(j, i, marker, ha='center', va='center', color='black')

plt.xticks(range(len(env_columns)), env_columns, rotation=45, ha='right')
plt.yticks(range(len(valid_species)), valid_species)
plt.title('Species-Environment Correlation Heatmap')
plt.tight_layout()
plt.savefig('../visualizations/cca_correlation_heatmap.png')
plt.close()

print("\nCCA analysis completed. Results saved to analysis_results/cca_results.txt")
print("Visualizations saved to visualizations/ directory")
