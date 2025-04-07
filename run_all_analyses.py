#!/usr/bin/env python3
"""
Unified Marine Species Analysis Script
-------------------------------------
This script runs all the statistical analyses on the marine species dataset:
1. Generalized Additive Models (GAMs)
2. Generalized Linear Mixed Models (GLMMs)
3. Canonical Correspondence Analysis (CCA)
4. Joint Species Distribution Models (JSDMs)
5. Time Series and State-Space Models
6. Non-Metric Multidimensional Scaling (NMDS)
7. Structural Equation Modeling (SEM)

Note: The Bubble species has been removed from all analyses as requested.
"""

import os
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directories if they don't exist
os.makedirs('analysis_results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Function to run a script and log its output
def run_script(script_name, log_file=None):
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(['python3', script_name], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True)
            
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                
            process.wait()
    else:
        process = subprocess.run(['python3', script_name], check=False)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nCompleted {script_name} in {duration:.2f} seconds")
    return process.returncode

# Function to check if visualization files exist
def check_visualizations(analysis_name, expected_files):
    print(f"\nChecking visualizations for {analysis_name}...")
    missing_files = []
    
    for file in expected_files:
        file_path = os.path.join('visualizations', file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} expected visualization files are missing:")
        for file in missing_files[:5]:  # Show first 5 missing files
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    else:
        print(f"All expected visualization files for {analysis_name} are present.")

# Function to check if result files exist
def check_results(analysis_name, expected_file):
    print(f"\nChecking results for {analysis_name}...")
    file_path = os.path.join('analysis_results', expected_file)
    
    if os.path.exists(file_path):
        print(f"Results file exists: {expected_file}")
        # Check file size
        size = os.path.getsize(file_path)
        if size < 100:
            print(f"Warning: Results file is very small ({size} bytes)")
    else:
        print(f"Warning: Results file is missing: {expected_file}")

# Load the dataset to get basic information
print("Loading dataset...")
data_path = 'upload/SHR_2022-2023_fauna_ctd_calcO2_pressure.csv'
data = pd.read_csv(data_path)

print(f"\nDataset Information:")
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")

# Print column names by category
species_columns = ['Sablefish', 'Sea Stars', 'Crabs', 'Hagfish', 
                  'Euphausia', 'Liponema', 'Flatfish', 'Rockfish', 'Eelpout']
env_columns = ['Temperature', 'Conductivity', 'Pressure', 'Salinity', 'Oxygen (ml/l)', 'PressurePSI']

print("\nSpecies columns (Bubble removed as requested):")
print(", ".join(species_columns))

print("\nEnvironmental columns:")
print(", ".join(env_columns))

# 1. Run GAM Analysis
gam_script = 'analysis_scripts/1_gam_analysis.py'
gam_log = 'analysis_results/gam_log.txt'
gam_status = run_script(gam_script, gam_log)

# Check GAM outputs
gam_viz_files = [f'gam_{species}_partial_effects.png' for species in species_columns]
gam_viz_files.append('gam_r2_comparison.png')
check_visualizations('GAM', gam_viz_files)
check_results('GAM', 'gam_results.txt')

# 2. Run GLMM Analysis (fixed version)
glmm_script = 'analysis_scripts/2_glmm_analysis_fixed.py'
glmm_log = 'analysis_results/glmm_log.txt'
glmm_status = run_script(glmm_script, glmm_log)

# Check GLMM outputs
glmm_viz_files = ['glmm_aic_comparison.png']
for species in species_columns:
    glmm_viz_files.append(f'glmm_{species}_coefficients.png')
    glmm_viz_files.append(f'glmm_{species}_predicted_vs_observed.png')
check_visualizations('GLMM', glmm_viz_files)
check_results('GLMM', 'glmm_results.txt')

# 3. Run CCA Analysis
cca_script = 'analysis_scripts/3_cca_analysis.py'
cca_log = 'analysis_results/cca_log.txt'
cca_status = run_script(cca_script, cca_log)

# Check CCA outputs
cca_viz_files = ['cca_biplot.png', 'cca_correlation_heatmap.png']
check_visualizations('CCA', cca_viz_files)
check_results('CCA', 'cca_results.txt')

# 4. Run JSDM Analysis
jsdm_script = 'analysis_scripts/4_jsdm_analysis.py'
jsdm_log = 'analysis_results/jsdm_log.txt'
jsdm_status = run_script(jsdm_script, jsdm_log)

# Check JSDM outputs
jsdm_viz_files = ['jsdm_cooccurrence_heatmap.png', 'jsdm_environment_correlations.png', 'jsdm_species_clusters.png']
check_visualizations('JSDM', jsdm_viz_files)
check_results('JSDM', 'jsdm_results.txt')

# 5. Run Time Series Analysis
ts_script = 'analysis_scripts/5_time_series_analysis.py'
ts_log = 'analysis_results/time_series_log.txt'
ts_status = run_script(ts_script, ts_log)

# Check Time Series outputs
ts_viz_files = ['time_series_var_impulse_response.png']
for env_var in ['Temperature', 'Salinity', 'Oxygen_ml_l', 'Pressure']:
    ts_viz_files.append(f'time_series_{env_var}_decomposition.png')
for species in species_columns:
    ts_viz_files.append(f'time_series_{species}_autocorrelation.png')
    ts_viz_files.append(f'time_series_{species}_sarimax.png')
check_visualizations('Time Series', ts_viz_files)
check_results('Time Series', 'time_series_results.txt')

# 6. Run NMDS Analysis
nmds_script = 'analysis_scripts/6_nmds_analysis.py'
nmds_log = 'analysis_results/nmds_log.txt'
nmds_status = run_script(nmds_script, nmds_log)

# Check NMDS outputs
nmds_viz_files = ['nmds_basic_plot.png', 'nmds_environmental_vectors.png', 
                 'nmds_by_month.png']
for env_var in ['Temperature', 'Salinity', 'Oxygen_ml_l', 'Pressure']:
    nmds_viz_files.append(f'nmds_{env_var}_gradient.png')
check_visualizations('NMDS', nmds_viz_files)
check_results('NMDS', 'nmds_results.txt')

# 7. Run SEM Analysis
sem_script = 'analysis_scripts/7_sem_analysis.py'
sem_log = 'analysis_results/sem_log.txt'
sem_status = run_script(sem_script, sem_log)

# Check SEM outputs
sem_viz_files = ['sem_path_diagram.png', 'sem_partial_correlations.png', 'sem_rsquared_comparison.png']
for species in ['Rockfish', 'Sea Stars', 'Sablefish']:
    sem_viz_files.append(f'sem_{species}_effects.png')
check_visualizations('SEM', sem_viz_files)
check_results('SEM', 'sem_results.txt')

# Summary of all analyses
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

analyses = [
    ("GAM", gam_status, "gam_results.txt"),
    ("GLMM", glmm_status, "glmm_results.txt"),
    ("CCA", cca_status, "cca_results.txt"),
    ("JSDM", jsdm_status, "jsdm_results.txt"),
    ("Time Series", ts_status, "time_series_results.txt"),
    ("NMDS", nmds_status, "nmds_results.txt"),
    ("SEM", sem_status, "sem_results.txt")
]

for name, status, result_file in analyses:
    status_text = "Success" if status == 0 else f"Failed (code {status})"
    file_exists = os.path.exists(os.path.join('analysis_results', result_file))
    file_status = "Results file exists" if file_exists else "Results file missing"
    print(f"{name:15} - {status_text:20} - {file_status}")

print("\nAll analyses completed. Results are available in the analysis_results directory.")
print("Visualizations are available in the visualizations directory.")
print("Open index.html to view the complete analysis report with all visualizations.")
print("\nNote: The Bubble species has been removed from all analyses as requested.")
