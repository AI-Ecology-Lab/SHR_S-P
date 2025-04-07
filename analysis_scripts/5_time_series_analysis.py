#!/usr/bin/env python3
"""
Time Series and State-Space Models
----------------------------------
Purpose: Investigate temporal dynamics and potential feedbacks in the system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.vector_ar.var_model import VAR
import os
from datetime import timedelta

# Create output directories if they don't exist
os.makedirs('../analysis_results', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('../SHR_2022-2023_fauna_ctd_calcO2_pressure.csv')

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.sort_values('Timestamp')  # Ensure data is sorted by time

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

if len(valid_species) < 1:
    print("Not enough variation in species data for time series analysis")
    with open('../analysis_results/time_series_results.txt', 'w') as f:
        f.write("Time series analysis could not be performed: Not enough variation in species data\n")
    exit()

print(f"Using {len(valid_species)} species with variation: {valid_species}")

# Create daily aggregated data for time series analysis
data['Date'] = data['Timestamp'].dt.date
daily_data = data.groupby('Date').agg({
    **{species: 'sum' for species in valid_species},
    **{env_var: 'mean' for env_var in env_columns}
}).reset_index()

# Convert Date back to datetime for time series analysis
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
daily_data = daily_data.set_index('Date')

# Check for sufficient time points
if len(daily_data) < 10:
    print("Not enough time points for meaningful time series analysis")
    with open('../analysis_results/time_series_results.txt', 'w') as f:
        f.write("Time series analysis could not be performed: Not enough time points\n")
    exit()

# 1. Time Series Decomposition for Environmental Variables
decomposition_results = {}

for env_var in env_columns:
    try:
        # Check if there's enough data and variation
        if len(daily_data[env_var].unique()) < 3:
            print(f"Not enough variation in {env_var} for decomposition")
            continue
            
        # Perform seasonal decomposition
        result = seasonal_decompose(daily_data[env_var], model='additive', period=min(7, len(daily_data)//2))
        decomposition_results[env_var] = result
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        result.observed.plot(ax=axes[0], title=f'{env_var} - Observed')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonal')
        result.resid.plot(ax=axes[3], title='Residual')
        plt.tight_layout()
        plt.savefig(f'../visualizations/time_series_{env_var}_decomposition.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in decomposition for {env_var}: {e}")

# 2. Autocorrelation Analysis for Species
for species in valid_species:
    try:
        # Check if there's enough non-zero data
        if daily_data[species].sum() < 10:
            print(f"Not enough non-zero data for {species} autocorrelation")
            continue
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot ACF
        plot_acf(daily_data[species], ax=axes[0], lags=min(20, len(daily_data)//2))
        axes[0].set_title(f'Autocorrelation Function for {species}')
        
        # Plot PACF
        plot_pacf(daily_data[species], ax=axes[1], lags=min(20, len(daily_data)//2))
        axes[1].set_title(f'Partial Autocorrelation Function for {species}')
        
        plt.tight_layout()
        plt.savefig(f'../visualizations/time_series_{species}_autocorrelation.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in autocorrelation analysis for {species}: {e}")

# 3. SARIMAX Models for Species with Environmental Covariates
sarimax_results = {}

for species in valid_species:
    try:
        # Check if there's enough non-zero data
        if daily_data[species].sum() < 10:
            print(f"Not enough non-zero data for {species} SARIMAX model")
            continue
            
        # Prepare exogenous variables (environmental factors)
        exog = daily_data[env_columns]
        
        # Fit SARIMAX model
        # Using simple AR(1) model with environmental covariates
        model = SARIMAX(daily_data[species], 
                        exog=exog,
                        order=(1, 0, 0),
                        enforce_stationarity=False)
        
        results = model.fit(disp=False)
        sarimax_results[species] = results
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot observed values
        ax.plot(daily_data.index, daily_data[species], 'o', label='Observed')
        
        # Plot fitted values
        ax.plot(daily_data.index, results.fittedvalues, 'r-', label='Fitted')
        
        ax.set_title(f'SARIMAX Model for {species}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'../visualizations/time_series_{species}_sarimax.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in SARIMAX model for {species}: {e}")

# 4. Vector Autoregression (VAR) for Species Interactions
try:
    # Check if we have enough species with variation
    if len(valid_species) >= 2:
        # Prepare data for VAR
        var_data = daily_data[valid_species].copy()
        
        # Fit VAR model
        var_model = VAR(var_data)
        var_results = var_model.fit(maxlags=min(5, len(daily_data)//10))
        
        # Plot impulse response functions
        fig, axes = plt.subplots(len(valid_species), len(valid_species), figsize=(15, 12))
        
        # Flatten axes if only one row
        if len(valid_species) == 1:
            axes = np.array([axes])
            
        # Generate impulse responses
        irf = var_results.irf(10)  # 10 periods ahead
        
        for i, response_var in enumerate(valid_species):
            for j, impulse_var in enumerate(valid_species):
                if len(valid_species) > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                
                ax.plot(range(len(irf.irfs[:, i, j])), irf.irfs[:, i, j])
                ax.set_title(f'Response of {response_var}\nto {impulse_var} Impulse')
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../visualizations/time_series_var_impulse_response.png')
        plt.close()
        
        # Granger causality tests
        granger_results = {}
        for i, species1 in enumerate(valid_species):
            granger_results[species1] = {}
            for j, species2 in enumerate(valid_species):
                if i != j:
                    try:
                        gc_res = var_results.test_causality(species1, [species2], kind='wald')
                        granger_results[species1][species2] = {
                            'test_statistic': gc_res.test_statistic,
                            'p_value': gc_res.pvalue,
                            'df': gc_res.df
                        }
                    except:
                        pass
    else:
        print("Not enough species with variation for VAR analysis")
        
except Exception as e:
    print(f"Error in VAR analysis: {e}")

# Save results
with open('../analysis_results/time_series_results.txt', 'w') as f:
    f.write("Time Series and State-Space Models Results\n")
    f.write("="*50 + "\n\n")
    
    # 1. Time Series Decomposition Results
    f.write("1. Time Series Decomposition for Environmental Variables\n")
    f.write("-"*50 + "\n")
    
    for env_var, result in decomposition_results.items():
        f.write(f"\n{env_var}:\n")
        f.write(f"  - Trend variance: {result.trend.var():.4f}\n")
        f.write(f"  - Seasonal variance: {result.seasonal.var():.4f}\n")
        f.write(f"  - Residual variance: {result.resid.var():.4f}\n")
        
        # Calculate variance explained
        total_var = result.observed.var()
        trend_pct = (result.trend.var() / total_var) * 100
        seasonal_pct = (result.seasonal.var() / total_var) * 100
        residual_pct = (result.resid.var() / total_var) * 100
        
        f.write(f"  - Variance explained by trend: {trend_pct:.2f}%\n")
        f.write(f"  - Variance explained by seasonality: {seasonal_pct:.2f}%\n")
        f.write(f"  - Variance explained by residuals: {residual_pct:.2f}%\n")
    
    # 2. SARIMAX Model Results
    f.write("\n\n2. SARIMAX Models for Species with Environmental Covariates\n")
    f.write("-"*60 + "\n")
    
    for species, result in sarimax_results.items():
        f.write(f"\n{species}:\n")
        f.write(f"  - AIC: {result.aic:.4f}\n")
        f.write(f"  - BIC: {result.bic:.4f}\n")
        f.write(f"  - Log Likelihood: {result.llf:.4f}\n\n")
        
        f.write("  Parameter Estimates:\n")
        for param, value in result.params.items():
            if param.startswith('x'):
                # This is an exogenous variable (environmental factor)
                env_idx = int(param[1:])
                if env_idx < len(env_columns):
                    env_name = env_columns[env_idx]
                    p_value = result.pvalues[param]
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    f.write(f"    - {env_name}: {value:.4f} (p = {p_value:.4f}) {significance}\n")
            else:
                # Other parameters (AR, MA, etc.)
                p_value = result.pvalues[param]
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                f.write(f"    - {param}: {value:.4f} (p = {p_value:.4f}) {significance}\n")
    
    # 3. VAR Model Results
    if 'var_results' in locals():
        f.write("\n\n3. Vector Autoregression (VAR) Results\n")
        f.write("-"*40 + "\n")
        
        f.write(f"\nModel Summary:\n")
        f.write(f"  - AIC: {var_results.aic:.4f}\n")
        f.write(f"  - BIC: {var_results.bic:.4f}\n")
        f.write(f"  - FPE: {var_results.fpe:.4f}\n")
        f.write(f"  - HQIC: {var_results.hqic:.4f}\n")
        
        f.write("\nGranger Causality Tests:\n")
        for species1, results in granger_results.items():
            for species2, res in results.items():
                significance = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
                f.write(f"  - {species1} Granger-causes {species2}: ")
                f.write(f"test statistic = {res['test_statistic']:.4f}, ")
                f.write(f"p-value = {res['p_value']:.4f} {significance}\n")

print("\nTime Series analysis completed. Results saved to analysis_results/time_series_results.txt")
print("Visualizations saved to visualizations/ directory")
