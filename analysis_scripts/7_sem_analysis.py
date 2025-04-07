#!/usr/bin/env python3
"""
Structural Equation Modeling (SEM)
----------------------------------
Purpose: Test hypotheses about causal relationships among multiple variables simultaneously.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
data['Hour'] = data['Timestamp'].dt.hour

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
    print("Not enough variation in species data for SEM analysis")
    with open('../analysis_results/sem_results.txt', 'w') as f:
        f.write("SEM analysis could not be performed: Not enough variation in species data\n")
    exit()

print(f"Using {len(valid_species)} species with variation: {valid_species}")

# Since we don't have a direct SEM implementation in Python without additional packages,
# we'll implement a simplified path analysis approach using multiple regression models

# Step 1: Check for multicollinearity in environmental variables
# Create a dataframe with environmental variables
env_data = data[env_columns].copy()
env_data = env_data.apply(lambda x: (x - x.mean()) / x.std())  # Standardize

# Calculate VIF (Variance Inflation Factor)
vif_data = pd.DataFrame()
vif_data["Variable"] = env_columns
vif_data["VIF"] = [variance_inflation_factor(env_data.values, i) for i in range(env_data.shape[1])]

# Identify variables with high multicollinearity (VIF > 10)
high_vif_vars = vif_data[vif_data["VIF"] > 10]["Variable"].tolist()
print(f"Variables with high multicollinearity (VIF > 10): {high_vif_vars}")

# If there's high multicollinearity, use PCA to create orthogonal environmental factors
if high_vif_vars:
    print("Using PCA to handle multicollinearity in environmental variables")
    pca = PCA()
    env_pca = pca.fit_transform(env_data)
    
    # Determine number of components to keep (explaining at least 90% variance)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    n_components = np.argmax(cumulative_variance >= 0.9) + 1
    
    # Create dataframe with PCA components
    env_pca_df = pd.DataFrame(
        env_pca[:, :n_components], 
        columns=[f"EnvPC{i+1}" for i in range(n_components)]
    )
    
    # Add PCA components to original data
    for col in env_pca_df.columns:
        data[col] = env_pca_df[col]
    
    # Use PCA components instead of original environmental variables
    env_predictors = env_pca_df.columns.tolist()
    
    # Save PCA loadings for interpretation
    pca_loadings = pd.DataFrame(
        pca.components_[:n_components, :],
        columns=env_columns,
        index=[f"EnvPC{i+1}" for i in range(n_components)]
    )
else:
    # Use original environmental variables
    env_predictors = env_columns

# Step 2: Define the path model structure
# For simplicity, we'll use a hierarchical structure:
# Environmental variables -> Species abundance

# Step 3: Fit regression models for each species
model_results = {}

for species in valid_species:
    print(f"\nFitting model for {species}...")
    
    # Convert to presence/absence for rare species
    if data[species].mean() < 0.1:
        print(f"Converting {species} to presence/absence due to low abundance")
        y = (data[species] > 0).astype(int)
        model_type = "logistic"
    else:
        y = data[species]
        model_type = "linear"
    
    # Fit model
    X = sm.add_constant(data[env_predictors])
    
    try:
        if model_type == "logistic":
            model = sm.Logit(y, X)
            result = model.fit(disp=0)
        else:
            model = sm.OLS(y, X)
            result = model.fit()
        
        # Store results
        model_results[species] = {
            'model_type': model_type,
            'result': result,
            'summary': result.summary(),
            'params': result.params,
            'pvalues': result.pvalues,
            'rsquared': result.prsquared if model_type == "logistic" else result.rsquared
        }
        
    except Exception as e:
        print(f"Error fitting model for {species}: {e}")

# Step 4: Calculate path coefficients and indirect effects
path_coefficients = {}
significant_paths = {}

for species, res in model_results.items():
    path_coefficients[species] = {}
    significant_paths[species] = []
    
    for predictor in env_predictors:
        if predictor in res['params']:
            coef = res['params'][predictor]
            p_val = res['pvalues'][predictor]
            
            path_coefficients[species][predictor] = {
                'coefficient': coef,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            
            if p_val < 0.05:
                significant_paths[species].append(predictor)

# Step 5: Calculate species correlations after controlling for environment
partial_correlations = {}

for i, sp1 in enumerate(valid_species):
    partial_correlations[sp1] = {}
    
    for j, sp2 in enumerate(valid_species):
        if i != j:
            # Calculate partial correlation controlling for environmental variables
            try:
                # Create dataframe with both species and environmental predictors
                partial_df = data[[sp1, sp2] + env_predictors].copy()
                
                # Fit models for each species with environmental predictors
                X_env = sm.add_constant(partial_df[env_predictors])
                
                # Residuals for sp1
                model1 = sm.OLS(partial_df[sp1], X_env)
                result1 = model1.fit()
                residuals1 = result1.resid
                
                # Residuals for sp2
                model2 = sm.OLS(partial_df[sp2], X_env)
                result2 = model2.fit()
                residuals2 = result2.resid
                
                # Calculate correlation between residuals
                corr, p_value = stats.pearsonr(residuals1, residuals2)
                
                partial_correlations[sp1][sp2] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
            except Exception as e:
                print(f"Error calculating partial correlation between {sp1} and {sp2}: {e}")
                partial_correlations[sp1][sp2] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'significant': False
                }

# Save results
with open('../analysis_results/sem_results.txt', 'w') as f:
    f.write("Structural Equation Modeling (SEM) Results\n")
    f.write("="*50 + "\n\n")
    
    # If PCA was used, save component loadings
    if 'pca_loadings' in locals():
        f.write("PCA Loadings for Environmental Variables\n")
        f.write("-"*40 + "\n")
        f.write(pca_loadings.to_string() + "\n\n")
        
        f.write("Explained Variance by PCA Components\n")
        f.write("-"*40 + "\n")
        for i in range(n_components):
            f.write(f"PC{i+1}: {explained_variance[i]:.4f} ({explained_variance[i]*100:.2f}%)\n")
        f.write(f"Cumulative (first {n_components} components): {cumulative_variance[n_components-1]:.4f} ({cumulative_variance[n_components-1]*100:.2f}%)\n\n")
    
    # Save model results for each species
    f.write("Model Results for Each Species\n")
    f.write("-"*30 + "\n")
    
    for species, res in model_results.items():
        f.write(f"\n{species} ({res['model_type']} model):\n")
        f.write(f"  - R² / Pseudo-R²: {res['rsquared']:.4f}\n")
        f.write("  - Significant predictors (p < 0.05):\n")
        
        if significant_paths[species]:
            for predictor in significant_paths[species]:
                coef = path_coefficients[species][predictor]['coefficient']
                p_val = path_coefficients[species][predictor]['p_value']
                f.write(f"    - {predictor}: coefficient = {coef:.4f} (p = {p_val:.4f})\n")
        else:
            f.write("    None\n")
    
    # Save partial correlations between species
    f.write("\n\nPartial Correlations Between Species (controlling for environment)\n")
    f.write("-"*60 + "\n")
    
    for sp1, correlations in partial_correlations.items():
        for sp2, corr_data in correlations.items():
            if corr_data['significant']:
                f.write(f"{sp1} <-> {sp2}: r = {corr_data['correlation']:.4f} (p = {corr_data['p_value']:.4f})\n")
    
    # Summary of key findings
    f.write("\n\nSummary of Key Findings\n")
    f.write("-"*25 + "\n")
    
    # Identify species most influenced by environment
    r2_values = [(species, res['rsquared']) for species, res in model_results.items()]
    r2_values.sort(key=lambda x: x[1], reverse=True)
    
    f.write("\nSpecies most influenced by environmental variables:\n")
    for species, r2 in r2_values[:3]:  # Top 3
        f.write(f"  - {species}: R² = {r2:.4f}\n")
    
    # Identify most important environmental predictors
    predictor_counts = {}
    for species, predictors in significant_paths.items():
        for pred in predictors:
            if pred not in predictor_counts:
                predictor_counts[pred] = 0
            predictor_counts[pred] += 1
    
    if predictor_counts:
        sorted_predictors = sorted(predictor_counts.items(), key=lambda x: x[1], reverse=True)
        
        f.write("\nMost influential environmental predictors:\n")
        for pred, count in sorted_predictors:
            f.write(f"  - {pred}: significant for {count} species\n")
    
    # Identify strongest species associations
    all_correlations = []
    for sp1, correlations in partial_correlations.items():
        for sp2, corr_data in correlations.items():
            if not np.isnan(corr_data['correlation']):
                all_correlations.append((sp1, sp2, corr_data['correlation'], corr_data['p_value']))
    
    if all_correlations:
        # Sort by absolute correlation strength
        all_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        f.write("\nStrongest species associations (after controlling for environment):\n")
        for sp1, sp2, corr, p_val in all_correlations[:5]:  # Top 5
            relation = "positive" if corr > 0 else "negative"
            significance = "significant" if p_val < 0.05 else "non-significant"
            f.write(f"  - {sp1} and {sp2}: {relation} association (r = {corr:.4f}, {significance})\n")

# Create visualizations

# 1. Path diagram (simplified version)
plt.figure(figsize=(14, 10))

# Set up positions
n_env = len(env_predictors)
n_species = len(valid_species)

env_y = 0.8
species_y = 0.2

env_x_positions = np.linspace(0.1, 0.9, n_env)
species_x_positions = np.linspace(0.1, 0.9, n_species)

# Plot nodes
for i, env in enumerate(env_predictors):
    plt.scatter(env_x_positions[i], env_y, s=300, color='skyblue', edgecolor='black', zorder=2)
    plt.text(env_x_positions[i], env_y, env, ha='center', va='center', fontsize=9)

for i, species in enumerate(valid_species):
    plt.scatter(species_x_positions[i], species_y, s=300, color='lightgreen', edgecolor='black', zorder=2)
    plt.text(species_x_positions[i], species_y, species, ha='center', va='center', fontsize=9)

# Plot significant paths
for i, species in enumerate(valid_species):
    for j, env in enumerate(env_predictors):
        if env in significant_paths[species]:
            coef = path_coefficients[species][env]['coefficient']
            
            # Determine line properties based on coefficient
            if coef > 0:
                color = 'red'
                linestyle = '-'
            else:
                color = 'blue'
                linestyle = '-'
            
            # Line width based on coefficient strength
            lw = abs(coef) * 2
            lw = max(0.5, min(lw, 4))  # Limit line width between 0.5 and 4
            
            # Draw the path
            plt.plot([env_x_positions[j], species_x_positions[i]], 
                    [env_y, species_y], 
                    color=color, linestyle=linestyle, linewidth=lw, alpha=0.7, zorder=1)

# Add legend
red_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='-')
blue_line = plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='-')
env_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', 
                      markeredgecolor='black', markersize=15)
species_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                          markeredgecolor='black', markersize=15)

plt.legend([env_point, species_point, red_line, blue_line], 
          ['Environmental Variable', 'Species', 'Positive Effect', 'Negative Effect'],
          loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)

plt.title('Simplified Path Diagram of Environmental Effects on Species')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('../visualizations/sem_path_diagram.png')
plt.close()

# 2. Heatmap of partial correlations between species
corr_matrix = np.zeros((len(valid_species), len(valid_species)))
p_value_matrix = np.zeros((len(valid_species), len(valid_species)))

for i, sp1 in enumerate(valid_species):
    for j, sp2 in enumerate(valid_species):
        if i != j:
            corr_matrix[i, j] = partial_correlations[sp1][sp2]['correlation']
            p_value_matrix[i, j] = partial_correlations[sp1][sp2]['p_value']
        else:
            corr_matrix[i, j] = 1.0  # Diagonal

plt.figure(figsize=(12, 10))
mask = np.zeros_like(corr_matrix, dtype=bool)
np.fill_diagonal(mask, True)  # Mask the diagonal

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
           xticklabels=valid_species, yticklabels=valid_species, mask=mask,
           vmin=-1, vmax=1)

# Add significance markers
for i in range(len(valid_species)):
    for j in range(len(valid_species)):
        if i != j and p_value_matrix[i, j] < 0.05:
            plt.text(j + 0.5, i + 0.5, '*', 
                    ha='center', va='center', color='black', fontsize=15)

plt.title('Partial Correlations Between Species (controlling for environment)')
plt.tight_layout()
plt.savefig('../visualizations/sem_partial_correlations.png')
plt.close()

# 3. Bar plot of R² values for each species
plt.figure(figsize=(12, 6))
species_names = [sp for sp, _ in r2_values]
r2_vals = [r2 for _, r2 in r2_values]

bars = plt.bar(species_names, r2_vals)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom')

plt.xlabel('Species')
plt.ylabel('R² / Pseudo-R²')
plt.title('Proportion of Variance Explained by Environmental Variables')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../visualizations/sem_rsquared_comparison.png')
plt.close()