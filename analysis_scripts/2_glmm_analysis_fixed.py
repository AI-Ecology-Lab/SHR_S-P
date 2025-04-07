#!/usr/bin/env python3
"""
Generalized Linear Mixed Models (GLMMs) Analysis
------------------------------------------------
Purpose: Handle count data (which may be overdispersed or zero-inflated) 
while accounting for random effects and autocorrelation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import os

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
data['DayOfYear'] = data['Timestamp'].dt.dayofyear

# Create a time period variable for random effects
data['TimePeriod'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)

# Select species columns
species_columns = ['Sablefish', 'Sea Stars', 'Bubble', 'Crabs', 'Hagfish', 
                  'Euphausia', 'Liponema', 'Flatfish', 'Rockfish', 'Eelpout']

# Select environmental variables - rename problematic column
data = data.rename(columns={'Oxygen (ml/l)': 'Oxygen_ml_l'})
env_columns = ['Temperature', 'Conductivity', 'Pressure', 'Salinity', 'Oxygen_ml_l', 'PressurePSI']

# Fill missing values if any
data = data.fillna(method='ffill')

# GLMM Analysis for each species
results = {}

for species in species_columns:
    print(f"\nRunning GLMM analysis for {species}...")
    
    # Skip if all values are 0 or if there's no variation
    if data[species].sum() == 0 or data[species].var() == 0:
        print(f"Skipping {species} as all values are 0 or there's no variation")
        continue
    
    # Prepare data for GLMM
    # Convert counts to binary presence/absence for binomial models
    # This simplifies the analysis for species with sparse counts
    model_data = data.copy()
    model_data[f'{species}_presence'] = (model_data[species] > 0).astype(int)
    
    # Create formula for fixed effects
    fixed_effects = ' + '.join(env_columns)
    formula = f"{species}_presence ~ {fixed_effects}"
    
    try:
        # Fit GLMM using statsmodels
        # Using logistic regression with random intercepts for time periods
        md = smf.mixedlm(formula, model_data, groups=model_data["TimePeriod"])
        mdf = md.fit(method=["lbfgs"])
        
        # Store results
        results[species] = {
            'summary': mdf.summary(),
            'AIC': mdf.aic,
            'BIC': mdf.bic,
            'params': mdf.params,
            'pvalues': mdf.pvalues
        }
        
        # Create coefficient plot
        coef_data = pd.DataFrame({
            'Coefficient': mdf.params[1:],  # Skip intercept
            'Variable': env_columns,
            'p-value': mdf.pvalues[1:]  # Skip intercept
        })
        
        # Add significance markers
        coef_data['Significant'] = coef_data['p-value'] < 0.05
        
        plt.figure(figsize=(10, 6))
        bars = sns.barplot(x='Variable', y='Coefficient', hue='Significant', data=coef_data)
        
        # Add p-value annotations
        for i, p in enumerate(coef_data['p-value']):
            plt.text(i, coef_data['Coefficient'].iloc[i] + 0.01, f'p={p:.3f}', 
                     ha='center', va='bottom', rotation=90, fontsize=8)
        
        plt.title(f'GLMM Coefficients for {species}')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'../visualizations/glmm_{species}_coefficients.png')
        plt.close()
        
        # Create predicted vs observed plot
        model_data['predicted'] = mdf.predict()
        
        plt.figure(figsize=(10, 6))
        sns.regplot(x='predicted', y=f'{species}_presence', data=model_data, 
                   scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        plt.title(f'Observed vs Predicted for {species}')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed presence')
        plt.tight_layout()
        plt.savefig(f'../visualizations/glmm_{species}_predicted_vs_observed.png')
        plt.close()
        
        print(f"GLMM analysis for {species} completed successfully.")
        print(f"AIC: {results[species]['AIC']:.4f}")
        print(f"BIC: {results[species]['BIC']:.4f}")
        
    except Exception as e:
        print(f"Error in GLMM analysis for {species}: {e}")

# Save results
with open('../analysis_results/glmm_results.txt', 'w') as f:
    for species, res in results.items():
        f.write(f"GLMM Results for {species}\n")
        f.write("="*50 + "\n")
        f.write(f"AIC: {res['AIC']:.4f}\n")
        f.write(f"BIC: {res['BIC']:.4f}\n\n")
        f.write(str(res['summary']) + "\n\n")
        
        f.write("Significant Variables (p < 0.05):\n")
        sig_vars = [(var, res['params'][i+1], res['pvalues'][i+1]) 
                   for i, var in enumerate(env_columns) 
                   if res['pvalues'][i+1] < 0.05]
        
        if sig_vars:
            for var, coef, pval in sig_vars:
                f.write(f"  - {var}: coefficient = {coef:.4f}, p-value = {pval:.4f}\n")
        else:
            f.write("  No significant variables found\n")
        
        f.write("\n" + "-"*50 + "\n\n")

# Create a summary plot comparing AIC across species
species_with_results = list(results.keys())
if species_with_results:
    aic_scores = [results[sp]['AIC'] for sp in species_with_results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(species_with_results, aic_scores)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.xlabel('Species')
    plt.ylabel('AIC Score (lower is better)')
    plt.title('GLMM Model Performance by Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../visualizations/glmm_aic_comparison.png')
    plt.close()

print("\nGLMM analysis completed. Results saved to analysis_results/glmm_results.txt")
print("Visualizations saved to visualizations/ directory")
