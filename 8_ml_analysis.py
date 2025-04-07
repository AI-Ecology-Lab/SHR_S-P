#!/usr/bin/env python3
"""
Machine Learning Approaches
--------------------------
Purpose: Uncover complex, nonlinear relationships and interactions in high-dimensional data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import os
import shap

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
data['Hour'] = data['Timestamp'].dt.hour
data['DayOfYear'] = data['Timestamp'].dt.dayofyear

# Select species columns
species_columns = ['Sablefish', 'Sea Stars', 'Bubble', 'Crabs', 'Hagfish', 
                  'Euphausia', 'Liponema', 'Flatfish', 'Rockfish', 'Eelpout']

# Select environmental variables
env_columns = ['Temperature', 'Conductivity', 'Pressure', 'Salinity', 'Oxygen (ml/l)', 'PressurePSI']

# Add temporal features
temporal_columns = ['Year', 'Month', 'Day', 'Hour', 'DayOfYear']

# Combine all predictor variables
predictor_columns = env_columns + temporal_columns

# Fill missing values if any
data = data.fillna(method='ffill')

# Check if there's enough variation in species data
species_var = data[species_columns].var()
valid_species = species_var[species_var > 0].index.tolist()

if len(valid_species) < 1:
    print("Not enough variation in species data for machine learning analysis")
    with open('../analysis_results/ml_results.txt', 'w') as f:
        f.write("Machine learning analysis could not be performed: Not enough variation in species data\n")
    exit()

print(f"Using {len(valid_species)} species with variation: {valid_species}")

# Prepare data for machine learning
X = data[predictor_columns].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=predictor_columns)

# Results storage
ml_results = {}

# 1. Random Forest Models for each species
for species in valid_species:
    print(f"\nRunning Random Forest analysis for {species}...")
    
    # Determine if we should use regression or classification
    # If species has very few non-zero values, use classification
    non_zero_ratio = (data[species] > 0).mean()
    
    if non_zero_ratio < 0.1:
        print(f"Using classification for {species} (presence/absence) due to low abundance")
        y = (data[species] > 0).astype(int)
        model_type = "classification"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_scaled_df, y, cv=5, scoring='accuracy')
        
        # Feature importance
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': predictor_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Store results
        ml_results[species] = {
            'model_type': model_type,
            'model': rf,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title(f'Top 10 Feature Importances for {species} (Random Forest Classification)')
        plt.tight_layout()
        plt.savefig(f'../visualizations/ml_{species}_rf_feature_importance.png')
        plt.close()
        
        # SHAP values for feature importance interpretation
        try:
            # Use a subset of data for SHAP analysis to reduce computation time
            X_sample = X_scaled_df.sample(min(1000, len(X_scaled_df)), random_state=42)
            
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[1], X_sample, feature_names=predictor_columns, show=False)
            plt.title(f'SHAP Feature Importance for {species} (Random Forest Classification)')
            plt.tight_layout()
            plt.savefig(f'../visualizations/ml_{species}_shap_summary.png')
            plt.close()
        except Exception as e:
            print(f"Error in SHAP analysis for {species}: {e}")
        
    else:
        print(f"Using regression for {species} (abundance)")
        y = data[species]
        model_type = "regression"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_scaled_df, y, cv=5, scoring='r2')
        
        # Feature importance
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': predictor_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Store results
        ml_results[species] = {
            'model_type': model_type,
            'model': rf,
            'mse': mse,
            'r2': r2,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title(f'Top 10 Feature Importances for {species} (Random Forest Regression)')
        plt.tight_layout()
        plt.savefig(f'../visualizations/ml_{species}_rf_feature_importance.png')
        plt.close()
        
        # Create actual vs predicted plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted Values for {species}')
        plt.tight_layout()
        plt.savefig(f'../visualizations/ml_{species}_actual_vs_predicted.png')
        plt.close()
        
        # SHAP values for feature importance interpretation
        try:
            # Use a subset of data for SHAP analysis to reduce computation time
            X_sample = X_scaled_df.sample(min(1000, len(X_scaled_df)), random_state=42)
            
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=predictor_columns, show=False)
            plt.title(f'SHAP Feature Importance for {species} (Random Forest Regression)')
            plt.tight_layout()
            plt.savefig(f'../visualizations/ml_{species}_shap_summary.png')
            plt.close()
        except Exception as e:
            print(f"Error in SHAP analysis for {species}: {e}")

# 2. Gradient Boosting for the most abundant species
# Find the most abundant species
most_abundant_species = max(valid_species, key=lambda x: data[x].sum())
print(f"\nRunning Gradient Boosting analysis for most abundant species: {most_abundant_species}")

y = data[most_abundant_species]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42
)

# Train Gradient Boosting regressor
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Evaluate model
y_pred = gb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Feature importance
importances = gb.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': predictor_columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Store results
ml_results[f"{most_abundant_species}_gb"] = {
    'model_type': 'regression',
    'model': gb,
    'mse': mse,
    'r2': r2,
    'feature_importance': feature_importance
}

# Create feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title(f'Top 10 Feature Importances for {most_abundant_species} (Gradient Boosting)')
plt.tight_layout()
plt.savefig(f'../visualizations/ml_{most_abundant_species}_gb_feature_importance.png')
plt.close()

# 3. Permutation Importance for more robust feature importance
for species in valid_species:
    if ml_results[species]['model_type'] == 'regression':
        model = ml_results[species]['model']
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        # Create dataframe
        perm_importance_df = pd.DataFrame({
            'Feature': predictor_columns,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Store results
        ml_results[species]['permutation_importance'] = perm_importance_df
        
        # Create permutation importance plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(10))
        plt.title(f'Top 10 Permutation Importances for {species}')
        plt.tight_layout()
        plt.savefig(f'../visualizations/ml_{species}_permutation_importance.png')
        plt.close()

# 4. Partial Dependence Plots for top features
for species in valid_species:
    model = ml_results[species]['model']
    
    # Get top 3 features
    top_features = ml_results[species]['feature_importance'].head(3)['Feature'].tolist()
    
    for feature in top_features:
        feature_idx = predictor_columns.index(feature)
        
        # Create feature values to evaluate
        feature_values = np.linspace(X_scaled_df[feature].min(), X_scaled_df[feature].max(), 100)
        
        # Create a dataset with all features at their mean except the target feature
        X_pdp = np.tile(X_scaled_df.mean().values, (len(feature_values), 1))
        
        # Vary the target feature
        X_pdp[:, feature_idx] = feature_values
        
        # Predict
        y_pdp = model.predict(X_pdp)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(feature_values, y_pdp)
        plt.xlabel(feature)
        plt.ylabel(f'Predicted {species}')
        plt.title(f'Partial Dependence Plot for {species} on {feature}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'../visualizations/ml_{species}_{feature}_partial_dependence.png')
        plt.close()

# Save results
with open('../analysis_results/ml_results.txt', 'w') as f:
    f.write("Machine Learning Analysis Results\n")
    f.write("="*50 + "\n\n")
    
    # Summary of model performance
    f.write("1. Model Performance Summary\n")
    f.write("-"*30 + "\n\n")
    
    for species, res in ml_results.items():
        if '_gb' not in species:  # Skip gradient boosting results for now
            f.write(f"{species} ({res['model_type']}):\n")
            
            if res['model_type'] == 'classification':
                f.write(f"  - Accuracy: {res['accuracy']:.4f}\n")
                f.write(f"  - Cross-validation accuracy: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}\n")
                
                # Add classification metrics
                f.write("  - Classification Report:\n")
                for label, metrics in res['classification_report'].items():
                    if label in ['0', '1']:
                        f.write(f"    Class {label}:\n")
                        f.write(f"      Precision: {metrics['precision']:.4f}\n")
                        f.write(f"      Recall: {metrics['recall']:.4f}\n")
                        f.write(f"      F1-score: {metrics['f1-score']:.4f}\n")
                
            else:  # regression
                f.write(f"  - Mean Squared Error: {res['mse']:.4f}\n")
                f.write(f"  - R² Score: {res['r2']:.4f}\n")
                f.write(f"  - Cross-validation R²: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}\n")
            
            f.write("\n")
    
    # Gradient Boosting results
    f.write("\n2. Gradient Boosting Results\n")
    f.write("-"*30 + "\n\n")
    
    for species, res in ml_results.items():
        if '_gb' in species:
            base_species = species.replace('_gb', '')
            f.write(f"Gradient Boosting for {base_species}:\n")
            f.write(f"  - Mean Squared Error: {res['mse']:.4f}\n")
            f.write(f"  - R² Score: {res['r2']:.4f}\n")
            
            # Compare with Random Forest
            rf_r2 = ml_results[base_species]['r2']
            f.write(f"  - Comparison with Random Forest: ")
            if res['r2'] > rf_r2:
                f.write(f"Gradient Boosting performs better (R² improvement: {res['r2'] - rf_r2:.4f})\n")
            else:
                f.write(f"Random Forest performs better (R² difference: {rf_r2 - res['r2']:.4f})\n")
            
            f.write("\n")
    
    # Feature importance summary
    f.write("\n3. Feature Importance Summary\n")
    f.write("-"*30 + "\n\n")
    
    # Aggregate feature importance across all species
    all_features = {}
    for feature in predictor_columns:
        all_features[feature] = 0
    
    for species, res in ml_results.items():
        if '_gb' not in species:  # Use only Random Forest results
            for _, row in res['feature_importance'].iterrows():
                feature = row['Feature']
                importance = row['Importance']
                all_features[feature] += importance
    
    # Normalize
    total_importance = sum(all_features.values())
    for feature in all_features:
        all_features[feature] /= total_importance
    
    # Sort and write
    sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
    
    f.write("Overall feature importance across all species:\n")
    for feature, importance in sorted_features:
        f.write(f"  - {feature}: {importance:.4f}\n")
    
    f.write("\nTop 3 features for each species:\n")
    for species, res in ml_results.items():
        if '_gb' not in species:  # Use only Random Forest results
            f.write(f"\n{species}:\n")
            top_features = res['feature_importance'].head(3)
        
(Content truncated due to size limit. Use line ranges to read in chunks)