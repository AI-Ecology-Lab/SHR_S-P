#!/usr/bin/env python3
"""
Generalized Additive Models (GAMs) Analysis
-------------------------------------------
Purpose: Capture nonlinear relationships between environmental predictors and species counts over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s, f
import os

# Create output directories if they don't exist
os.makedirs('../analysis_results', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('../SHR_2022-2023_fauna_ctd_calcO2_pressure.csv')

# Basic data exploration
print(f"Dataset shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour

# Select species columns
species_columns = ['Sablefish', 'Sea Stars', 'Bubble', 'Crabs', 'Hagfish', 
                  'Euphausia', 'Liponema', 'Flatfish', 'Rockfish', 'Eelpout']

# Select environmental variables
env_columns = ['Temperature', 'Conductivity', 'Pressure', 'Salinity', 'Oxygen (ml/l)', 'PressurePSI']

# Summary statistics
print("\nSummary statistics for species counts:")
print(data[species_columns].describe())

print("\nSummary statistics for environmental variables:")
print(data[env_columns].describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Fill missing values if any
data = data.fillna(method='ffill')

# GAM Analysis for each species
results = {}

for species in species_columns:
    print(f"\nRunning GAM analysis for {species}...")
    
    # Skip if all values are 0
    if data[species].sum() == 0:
        print(f"Skipping {species} as all values are 0")
        continue
    
    # Prepare data for GAM
    X = data[env_columns].values
    y = data[species].values
    
    # Fit GAM model
    try:
        gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5))
        gam.fit(X, y)
        
        # Store results
        results[species] = {
            'model': gam,
            'summary': gam.summary(),
            'r2_score': gam.statistics_['pseudo_r2']['explained_deviance'],
            'AIC': gam.statistics_['AIC'],
            'n_splines': gam.n_splines
        }
        
        # Create partial dependence plots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        
        for i, feature in enumerate(env_columns):
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            
            axs[i].plot(XX[:, i], pdep)
            axs[i].fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.2)
            axs[i].set_title(f'Partial Dependence: {feature}')
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel(f'Partial effect on {species} count')
        
        plt.tight_layout()
        plt.savefig(f'../visualizations/gam_{species}_partial_effects.png')
        plt.close()
        
        print(f"GAM analysis for {species} completed successfully.")
        print(f"R² score: {results[species]['r2_score']:.4f}")
        print(f"AIC: {results[species]['AIC']:.4f}")
        
    except Exception as e:
        print(f"Error in GAM analysis for {species}: {e}")

# Save results
with open('../analysis_results/gam_results.txt', 'w') as f:
    for species, res in results.items():
        f.write(f"GAM Results for {species}\n")
        f.write("="*50 + "\n")
        f.write(f"R² score: {res['r2_score']:.4f}\n")
        f.write(f"AIC: {res['AIC']:.4f}\n")
        f.write(f"Number of splines: {res['n_splines']}\n\n")
        f.write(str(res['summary']) + "\n\n")

# Create a summary plot
species_with_results = list(results.keys())
if species_with_results:
    r2_scores = [results[sp]['r2_score'] for sp in species_with_results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(species_with_results, r2_scores)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Species')
    plt.ylabel('R² Score')
    plt.title('GAM Model Performance by Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../visualizations/gam_r2_comparison.png')
    plt.close()

print("\nGAM analysis completed. Results saved to analysis_results/gam_results.txt")
print("Visualizations saved to visualizations/ directory")
