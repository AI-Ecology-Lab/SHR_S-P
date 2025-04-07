#!/usr/bin/env python3
"""
Wavelet Analysis
---------------
Purpose: Identify periodic patterns and changes in variability over different time scales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy import signal
import os
from datetime import timedelta

# Create output directories if they don't exist
os.makedirs('../analysis_results', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('../upload/SHR_2022-2023_fauna_ctd_calcO2_pressure.csv')

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
    print("Not enough variation in species data for wavelet analysis")
    with open('../analysis_results/wavelet_results.txt', 'w') as f:
        f.write("Wavelet analysis could not be performed: Not enough variation in species data\n")
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
if len(daily_data) < 20:
    print("Not enough time points for meaningful wavelet analysis")
    with open('../analysis_results/wavelet_results.txt', 'w') as f:
        f.write("Wavelet analysis could not be performed: Not enough time points\n")
    exit()

# Ensure the time series is evenly spaced
# If there are gaps, we'll need to resample or interpolate
date_diffs = daily_data.index.to_series().diff().dropna()
if date_diffs.nunique() > 1:
    print("Time series has gaps. Resampling to daily frequency...")
    
    # Create a complete date range
    date_range = pd.date_range(start=daily_data.index.min(), end=daily_data.index.max(), freq='D')
    
    # Reindex and interpolate
    daily_data = daily_data.reindex(date_range)
    daily_data = daily_data.interpolate(method='linear')

# Function to perform wavelet analysis
def perform_wavelet_analysis(time_series, title, output_file, scales=None):
    # Remove mean and normalize by standard deviation
    normalized_series = (time_series - time_series.mean()) / time_series.std()
    
    # Set default scales if not provided
    if scales is None:
        scales = np.arange(1, min(128, len(normalized_series)//2))
    
    # Choose wavelet function
    wavelet = 'morl'  # Morlet wavelet
    
    # Perform continuous wavelet transform
    coefficients, frequencies = pywt.cwt(normalized_series, scales, wavelet)
    
    # Convert scales to periods (in days)
    periods = 1.0 / frequencies
    
    # Create time array
    time_array = np.arange(len(normalized_series))
    
    # Calculate power (squared absolute value of coefficients)
    power = (abs(coefficients)) ** 2
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot original time series
    ax1.plot(time_array, normalized_series)
    ax1.set_title(f'Normalized Time Series: {title}')
    ax1.set_ylabel('Normalized Value')
    
    # Plot wavelet power spectrum
    contour_levels = np.linspace(0, np.max(power), 100)
    contourf = ax2.contourf(time_array, np.log2(periods), power, contour_levels, 
                          extend='both', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax2)
    cbar.set_label('Wavelet Power')
    
    # Set y-axis (period) ticks
    period_ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    period_ticks = [tick for tick in period_ticks if tick <= max(periods)]
    ax2.set_yticks(np.log2(period_ticks))
    ax2.set_yticklabels(period_ticks)
    
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Period (days)')
    ax2.set_title('Wavelet Power Spectrum')
    
    # Add cone of influence
    # The edge effects become important where the wavelet power drops by a factor of e^-2
    coi_scale = scales[-1]
    time_max = len(normalized_series) - 1
    coi = np.zeros(len(normalized_series))
    for i in range(len(normalized_series)):
        if i < coi_scale:
            coi[i] = i
        elif i >= time_max - coi_scale:
            coi[i] = time_max - i
        else:
            coi[i] = coi_scale
    
    # Plot cone of influence
    ax2.plot(time_array, np.log2(coi), 'k--')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return {
        'power': power,
        'periods': periods,
        'coefficients': coefficients,
        'max_power_period': periods[np.argmax(np.mean(power, axis=1))]
    }

# Perform wavelet analysis for environmental variables
env_wavelet_results = {}

for env_var in env_columns:
    print(f"Performing wavelet analysis for {env_var}...")
    
    # Check if there's enough variation
    if daily_data[env_var].std() < 1e-6:
        print(f"Not enough variation in {env_var} for wavelet analysis")
        continue
    
    try:
        result = perform_wavelet_analysis(
            daily_data[env_var].values,
            env_var,
            f'../visualizations/wavelet_{env_var}.png'
        )
        
        env_wavelet_results[env_var] = result
        
    except Exception as e:
        print(f"Error in wavelet analysis for {env_var}: {e}")

# Perform wavelet analysis for species
species_wavelet_results = {}

for species in valid_species:
    print(f"Performing wavelet analysis for {species}...")
    
    # Check if there's enough non-zero data
    if (daily_data[species] > 0).sum() < 10:
        print(f"Not enough non-zero data for {species} wavelet analysis")
        continue
    
    try:
        result = perform_wavelet_analysis(
            daily_data[species].values,
            species,
            f'../visualizations/wavelet_{species}.png'
        )
        
        species_wavelet_results[species] = result
        
    except Exception as e:
        print(f"Error in wavelet analysis for {species}: {e}")

# Cross-wavelet analysis between species and environmental variables
cross_wavelet_results = {}

# Function to perform cross-wavelet analysis
def perform_cross_wavelet_analysis(series1, series2, title1, title2, output_file, scales=None):
    # Remove mean and normalize by standard deviation
    norm_series1 = (series1 - series1.mean()) / series1.std()
    norm_series2 = (series2 - series2.mean()) / series2.std()
    
    # Set default scales if not provided
    if scales is None:
        scales = np.arange(1, min(128, len(norm_series1)//2))
    
    # Choose wavelet function
    wavelet = 'morl'  # Morlet wavelet
    
    # Perform continuous wavelet transform for both series
    coeffs1, freqs = pywt.cwt(norm_series1, scales, wavelet)
    coeffs2, _ = pywt.cwt(norm_series2, scales, wavelet)
    
    # Calculate cross-wavelet transform
    cross_wavelet = coeffs1 * np.conj(coeffs2)
    
    # Calculate power and phase
    power = np.abs(cross_wavelet) ** 2
    phase = np.angle(cross_wavelet)
    
    # Convert scales to periods (in days)
    periods = 1.0 / freqs
    
    # Create time array
    time_array = np.arange(len(norm_series1))
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot original time series
    ax1.plot(time_array, norm_series1, 'b-', label=title1)
    ax1.plot(time_array, norm_series2, 'r-', label=title2)
    ax1.set_title(f'Normalized Time Series: {title1} vs {title2}')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    
    # Plot cross-wavelet power spectrum
    contour_levels = np.linspace(0, np.max(power), 100)
    contourf = ax2.contourf(time_array, np.log2(periods), power, contour_levels, 
                          extend='both', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax2)
    cbar.set_label('Cross-Wavelet Power')
    
    # Set y-axis (period) ticks
    period_ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    period_ticks = [tick for tick in period_ticks if tick <= max(periods)]
    ax2.set_yticks(np.log2(period_ticks))
    ax2.set_yticklabels(period_ticks)
    
    ax2.set_ylabel('Period (days)')
    ax2.set_title('Cross-Wavelet Power Spectrum')
    
    # Plot phase difference
    phase_contourf = ax3.contourf(time_array, np.log2(periods), phase, 
                                np.linspace(-np.pi, np.pi, 100), 
                                extend='both', cmap='hsv')
    
    # Add colorbar
    cbar_phase = plt.colorbar(phase_contourf, ax=ax3)
    cbar_phase.set_label('Phase Difference (radians)')
    
    # Set y-axis (period) ticks
    ax3.set_yticks(np.log2(period_ticks))
    ax3.set_yticklabels(period_ticks)
    
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Period (days)')
    ax3.set_title('Phase Difference')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return {
        'power': power,
        'phase': phase,
        'periods': periods,
        'max_power_period': periods[np.argmax(np.mean(power, axis=1))],
        'mean_phase': np.mean(phase)
    }

# Perform cross-wavelet analysis for selected species-environment pairs
# Choose the most abundant species and most variable environmental factors
if valid_species and env_columns:
    most_abundant_species = max(valid_species, key=lambda x: daily_data[x].sum())
    
    # Choose top 2 environmental variables with highest variance
    env_vars_by_variance = sorted(env_columns, key=lambda x: daily_data[x].var(), reverse=True)[:2]
    
    for env_var in env_vars_by_variance:
        pair_key = f"{most_abundant_species}_{env_var}"
        print(f"Performing cross-wavelet analysis for {pair_key}...")
        
        try:
            result = perform_cross_wavelet_analysis(
                daily_data[most_abundant_species].values,
                daily_data[env_var].values,
                most_abundant_species,
                env_var,
                f'../visualizations/cross_wavelet_{pair_key}.png'
            )
            
            cross_wavelet_results[pair_key] = result
            
        except Exception as e:
            print(f"Error in cross-wavelet analysis for {pair_key}: {e}")

# Save results
with open('../analysis_results/wavelet_results.txt', 'w') as f:
    f.write("Wavelet Analysis Results\n")
    f.write("="*50 + "\n\n")
    
    # Environmental variables wavelet results
    f.write("1. Environmental Variables Wavelet Analysis\n")
    f.write("-"*40 + "\n\n")
    
    for env_var, result in env_wavelet_results.items():
        f.write(f"{env_var}:\n")
        f.write(f"  - Dominant period: {result['max_power_period']:.2f} days\n")
        
        # Identify periods with high power
        mean_power = np.mean(result['power'], axis=1)
        high_power_indices = np.where(mean_power > 0.5 * np.max(mean_power))[0]
        high_power_periods = result['periods'][high_power_indices]
        
        if len(high_power_periods) > 0:
            f.write("  - Periods with high power:\n")
            for period in high_power_periods:
                f.write(f"    * {period:.2f} days\n")
        
        f.write("\n")
    
    # Species wavelet results
    f.write("\n2. Species Wavelet Analysis\n")
    f.write("-"*30 + "\n\n")
    
    for species, result in species_wavelet_results.items():
        f.write(f"{species}:\n")
        f.write(f"  - Dominant period: {result['max_power_period']:.2f} days\n")
        
        # Identify periods with high power
        mean_power = np.mean(result['power'], axis=1)
        high_power_indices = np.where(mean_power > 0.5 * np.max(mean_power))[0]
        high_power_periods = result['periods'][high_power_indices]
        
        if len(high_power_periods) > 0:
            f.write("  - Periods with high power:\n")
            for period in high_power_periods:
                f.write(f"    * {period:.2f} days\n")
        
        f.write("\n")
    
    # Cross-wavelet results
    f.write("\n3. Cross-Wavelet Analysis\n")
    f.write("-"*30 + "\n\n")
    
    for pair_key, result in cross_wavelet_results.items():
        species, env_var = pair_key.split('_', 1)
        f.write(f"{species} vs {env_var}:\n")
        f.write(f"  - Dominant period of coherence: {result['max_power_period']:.2f} days\n")
        
        # Interpret phase difference
        mean_phase = result['mean_phase']
        phase_interpretation = ""
        if -np.pi/4 <= mean_phase <= np.pi/4:
            phase_interpretation = "in phase (positively correlated)"
        elif np.pi/4 < mean_phase <= 3*np.pi/4:
            phase_interpretation = "species leads environmental variable by 90°"
        elif mean_phase > 3*np.pi/4 or mean_phase < -3*np.pi/4:
            phase_interpretation = "anti-phase (negatively correlated)"
        else:  # -3*np.pi/4 <= mean_phase < -np.pi/4
            phase_interpretation = "environmental variable leads species by 90°"
        
        f.write(f"  - Mean phase difference: {mean_phase:.2f} radians ({phase_interpretation})\n")
        
        # Identify periods with high coherence
        mean_power = np.mean(result['power'], axis=1)
        high_power_indices = np.where(mean_power > 0.5 * np.max(mean_power))[0]
        high_power_periods = result['periods'][high_power_indices]
        
        if len(high_power_periods) > 0:
            f.write("  - Periods with high coherence:\n")
            for period in high_power_periods:
                f.write(f"    * {period:.2f} days\n")
        
        f.write("\n")
    
    # Summary of findings
    f.write("\n4. Summary of Findings\n")
    f.write("-"*25 + "\n\n")
    
    # Identify common periodicities across variables
    all_dominant_periods = []
    for env_var, result in env_wavelet_results.items():
        all_dominant_periods.append((env_var, result['max_power_period'], 'environmental'))
    
    for species, result in species_wavelet_results.items():
        all_dominant_periods.append((species, result['max_power_period'], 'species'))
    
    # Group similar periods (within 10% of each other)
    grouped_periods = []
    for name, period, var_type in all_dominant_periods:
        found_group = False
        for group in grouped_periods:
            if any(abs(period - p) / p < 0.1 for _, p, _ in group):
                group.append((name, period, var_type))
                found_group = True
                break
        
        if not found_group:
            grouped_periods.append([(name, period, var_type)])
    
    # Write common periodicities
    f.write("Common periodicities across variables:\n")
    for i, group in enumerate(grouped_periods):
        if len(group) > 1:
            avg_period = np.mean([p for _, p, _ in group])
            f.write(f"\nGroup {i+1} (around {avg_period:.2f} days):\n")
            
            for name, period, var_type in group:
                f.write(f"  - {name} ({var_type}): {period:.2f} days\n")
    
    # Identify potential environmental drivers based on cross-wavelet analysis
    if cross_wavelet_results:
        f.write("\nPotential environmental drivers based on cross-wavelet analysis:\n")

(Content truncated due to size limit. Use line ranges to read in chunks)