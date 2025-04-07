#!/usr/bin/env python3
"""
Non-Metric Multidimensional Scaling (NMDS)
------------------------------------------
Purpose: Visualize overall patterns in community structure in a reduced-dimensional space.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import os
from scipy import stats

# Create output directories if they don't exist
os.makedirs('../analysis_results', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('../upload/SHR_2022-2023_fauna_ctd_calcO2_pressure.csv')

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
    print("Not enough variation in species data for NMDS analysis")
    with open('../analysis_results/nmds_results.txt', 'w') as f:
        f.write("NMDS analysis could not be performed: Not enough variation in species data\n")
    exit()

print(f"Using {len(valid_species)} species with variation: {valid_species}")

# Extract species data
species_data = data[valid_species].values

# Calculate distance matrix (Bray-Curtis dissimilarity for ecological data)
def bray_curtis(X):
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Bray-Curtis dissimilarity
            numerator = np.sum(np.abs(X[i] - X[j]))
            denominator = np.sum(X[i] + X[j])
            
            # Avoid division by zero
            if denominator > 0:
                dist_matrix[i, j] = numerator / denominator
            else:
                dist_matrix[i, j] = 0
                
            # Distance matrix is symmetric
            dist_matrix[j, i] = dist_matrix[i, j]
            
    return dist_matrix

# For large datasets, sample a subset of data points to make NMDS computationally feasible
max_samples = 1000
if len(data) > max_samples:
    print(f"Sampling {max_samples} data points for NMDS analysis")
    sample_indices = np.random.choice(len(data), max_samples, replace=False)
    species_subset = species_data[sample_indices]
    env_subset = data[env_columns].values[sample_indices]
    timestamps = data['Timestamp'].iloc[sample_indices]
else:
    species_subset = species_data
    env_subset = data[env_columns].values
    timestamps = data['Timestamp']

# Calculate distance matrix
print("Calculating distance matrix...")
try:
    distances = bray_curtis(species_subset)
except Exception as e:
    print(f"Error calculating Bray-Curtis distances: {e}")
    print("Using Euclidean distances instead")
    distances = pairwise_distances(species_subset, metric='euclidean')

# Perform NMDS
print("Performing NMDS...")
nmds = MDS(n_components=2, metric=False, dissimilarity='precomputed', 
          random_state=42, n_init=10, max_iter=300)

try:
    nmds_result = nmds.fit_transform(distances)
    stress = nmds.stress_
    
    print(f"NMDS completed with stress value: {stress:.4f}")
    
    # Create a dataframe with NMDS coordinates
    nmds_df = pd.DataFrame({
        'NMDS1': nmds_result[:, 0],
        'NMDS2': nmds_result[:, 1],
        'Timestamp': timestamps
    })
    
    # Add environmental variables
    for i, col in enumerate(env_columns):
        nmds_df[col] = env_subset[:, i]
    
    # Add month and year information
    nmds_df['Month'] = pd.to_datetime(nmds_df['Timestamp']).dt.month
    nmds_df['Year'] = pd.to_datetime(nmds_df['Timestamp']).dt.year
    
    # Calculate correlations between NMDS axes and environmental variables
    env_correlations = {}
    for env_var in env_columns:
        corr_axis1, p_val1 = stats.pearsonr(nmds_df['NMDS1'], nmds_df[env_var])
        corr_axis2, p_val2 = stats.pearsonr(nmds_df['NMDS2'], nmds_df[env_var])
        
        env_correlations[env_var] = {
            'NMDS1': {'correlation': corr_axis1, 'p_value': p_val1},
            'NMDS2': {'correlation': corr_axis2, 'p_value': p_val2}
        }
    
    # Save results
    with open('../analysis_results/nmds_results.txt', 'w') as f:
        f.write("Non-Metric Multidimensional Scaling (NMDS) Results\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"NMDS Stress Value: {stress:.4f}\n")
        f.write("Interpretation of stress value:\n")
        if stress < 0.05:
            f.write("  - Excellent representation (stress < 0.05)\n")
        elif stress < 0.1:
            f.write("  - Good representation (stress < 0.1)\n")
        elif stress < 0.2:
            f.write("  - Fair representation (stress < 0.2)\n")
        else:
            f.write("  - Poor representation (stress > 0.2)\n")
        
        f.write("\nCorrelations between NMDS axes and environmental variables:\n")
        f.write("-"*60 + "\n")
        
        for env_var, corrs in env_correlations.items():
            f.write(f"\n{env_var}:\n")
            
            # NMDS1 correlation
            corr1 = corrs['NMDS1']['correlation']
            p_val1 = corrs['NMDS1']['p_value']
            significance1 = "***" if p_val1 < 0.001 else "**" if p_val1 < 0.01 else "*" if p_val1 < 0.05 else ""
            f.write(f"  - NMDS1: r = {corr1:.4f} (p = {p_val1:.4f}) {significance1}\n")
            
            # NMDS2 correlation
            corr2 = corrs['NMDS2']['correlation']
            p_val2 = corrs['NMDS2']['p_value']
            significance2 = "***" if p_val2 < 0.001 else "**" if p_val2 < 0.01 else "*" if p_val2 < 0.05 else ""
            f.write(f"  - NMDS2: r = {corr2:.4f} (p = {p_val2:.4f}) {significance2}\n")
    
    # Create visualizations
    
    # 1. Basic NMDS plot
    plt.figure(figsize=(10, 8))
    plt.scatter(nmds_df['NMDS1'], nmds_df['NMDS2'], alpha=0.6)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.title('NMDS Ordination of Species Composition')
    plt.xlabel('NMDS1')
    plt.ylabel('NMDS2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/nmds_basic_plot.png')
    plt.close()
    
    # 2. NMDS plot colored by month
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(nmds_df['NMDS1'], nmds_df['NMDS2'], 
                         c=nmds_df['Month'], cmap='viridis', 
                         alpha=0.7, s=50)
    plt.colorbar(scatter, label='Month')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.title('NMDS Ordination by Month')
    plt.xlabel('NMDS1')
    plt.ylabel('NMDS2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/nmds_by_month.png')
    plt.close()
    
    # 3. NMDS plot with environmental vectors
    # Only plot significant correlations (p < 0.05)
    plt.figure(figsize=(12, 10))
    plt.scatter(nmds_df['NMDS1'], nmds_df['NMDS2'], alpha=0.5, color='gray')
    
    # Add environmental vectors
    for env_var, corrs in env_correlations.items():
        corr1 = corrs['NMDS1']['correlation']
        corr2 = corrs['NMDS2']['correlation']
        p_val1 = corrs['NMDS1']['p_value']
        p_val2 = corrs['NMDS2']['p_value']
        
        # Only plot if at least one correlation is significant
        if p_val1 < 0.05 or p_val2 < 0.05:
            # Scale vector for visibility
            scaling_factor = 1.0
            plt.arrow(0, 0, corr1 * scaling_factor, corr2 * scaling_factor, 
                     head_width=0.05, head_length=0.1, fc='red', ec='red')
            plt.text(corr1 * scaling_factor * 1.1, corr2 * scaling_factor * 1.1, 
                    env_var, color='red', ha='center', va='center')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.title('NMDS Ordination with Environmental Vectors')
    plt.xlabel('NMDS1')
    plt.ylabel('NMDS2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/nmds_environmental_vectors.png')
    plt.close()
    
    # 4. NMDS plot with environmental gradient
    # Choose the environmental variable with the strongest correlation
    strongest_var = max(env_correlations.items(), 
                       key=lambda x: max(abs(x[1]['NMDS1']['correlation']), 
                                        abs(x[1]['NMDS2']['correlation'])))[0]
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(nmds_df['NMDS1'], nmds_df['NMDS2'], 
                         c=nmds_df[strongest_var], cmap='coolwarm', 
                         alpha=0.7, s=50)
    plt.colorbar(scatter, label=strongest_var)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.title(f'NMDS Ordination with {strongest_var} Gradient')
    plt.xlabel('NMDS1')
    plt.ylabel('NMDS2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../visualizations/nmds_{strongest_var}_gradient.png')
    plt.close()
    
    print("\nNMDS analysis completed. Results saved to analysis_results/nmds_results.txt")
    print("Visualizations saved to visualizations/ directory")
    
except Exception as e:
    print(f"Error in NMDS analysis: {e}")
    with open('../analysis_results/nmds_results.txt', 'w') as f:
        f.write(f"NMDS analysis failed with error: {e}\n")
