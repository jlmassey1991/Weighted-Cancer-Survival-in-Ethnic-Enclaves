"""
Ethnic Enclave Cancer Survival Analysis
=======================================

This script performs survival analysis examining the impact of ethnic enclave residence 
on cancer survival outcomes across different cancer types and racial/ethnic groups.

Key Features:
- Inverse Probability Weighting (IPW) for treatment group balancing
- Kaplan-Meier survival estimation with confidence intervals
- Proportional hazards modeling with realistic effect sizes
- Visualization of survival curves stratified by enclave residence

Dependencies: pandas, numpy, matplotlib, seaborn, lifelines, scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_survival_data(n=8000):
    """
    Generate realistic cancer survival data with ethnic enclave exposure.
    
    Parameters:
    -----------
    n : int
        Total sample size
        
    Returns:
    --------
    pd.DataFrame
        Generated survival dataset
    """
    # Define categories
    cancer_types = ['breast', 'lung', 'colon', 'cervical']
    races = ['hispanic', 'black', 'asian', 'white']
    enclave_types = ['Dense Enclave', 'Sparse Enclave', 'No Enclave']
    
    data = []
    
    for i in range(n):
        # Randomly assign basic characteristics
        race = np.random.choice(races)
        cancer = np.random.choice(cancer_types)
        enclave = np.random.choice(enclave_types)
        
        # Generate covariates with realistic correlations
        age = np.clip(np.random.normal(60, 15), 18, 95)
        stage = np.random.choice(['Early', 'Local', 'Advanced'], p=[0.3, 0.4, 0.3])
        ses = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.35, 0.25])
        insurance = np.random.choice(['Uninsured', 'Public', 'Private'], p=[0.15, 0.45, 0.4])
        
        # Calculate propensity scores for IPW
        propensity_score = 0.33  # Base probability
        if race in ['hispanic', 'asian']:
            propensity_score += 0.2
        if ses == 'Low':
            propensity_score += 0.15
        if insurance == 'Uninsured':
            propensity_score += 0.1
            
        # Generate survival times with complex interactions
        hazard_ratio = 1.0
        
        # Race-enclave specific effects with dramatic separations
        if race == 'black':
            # Worse outcomes across all cancers in enclaves
            if enclave == 'Dense Enclave':
                hazard_ratio *= 2.20  # Much worse survival
            elif enclave == 'Sparse Enclave':
                hazard_ratio *= 1.45
        elif race == 'hispanic' and cancer in ['colon', 'cervical']:
            # Very strong protective effects
            if enclave == 'Dense Enclave':
                hazard_ratio *= 0.45  # Much better survival
            elif enclave == 'Sparse Enclave':
                hazard_ratio *= 0.70
        elif race == 'asian' and cancer == 'lung':
            # Very strong protective effects
            if enclave == 'Dense Enclave':
                hazard_ratio *= 0.40  # Much better survival
            elif enclave == 'Sparse Enclave':
                hazard_ratio *= 0.65
        else:
            # Moderate protective effects for other combinations
            if enclave == 'Dense Enclave':
                hazard_ratio *= 0.75
            elif enclave == 'Sparse Enclave':
                hazard_ratio *= 0.85
        
        # Adjust for other clinical factors
        if stage == 'Advanced':
            hazard_ratio *= 2.5
        elif stage == 'Local':
            hazard_ratio *= 1.4
        if age > 70:
            hazard_ratio *= 1.3
        if ses == 'Low':
            hazard_ratio *= 1.25
        if insurance == 'Uninsured':
            hazard_ratio *= 1.35
            
        # Generate survival time (exponential distribution)
        base_rate = {'lung': 0.08, 'cervical': 0.06, 'breast': 0.04, 'colon': 0.04}[cancer]
        survival_time = -np.log(np.random.random()) / (base_rate * hazard_ratio)
        
        # Apply censoring
        max_followup = 60  # 5 years in months
        censored = (survival_time > max_followup) or (np.random.random() < 0.3)
        survival_time = min(survival_time, max_followup)
        
        # Calculate IPW weights
        actual_enclave = 1 if enclave == 'Dense Enclave' else (0.5 if enclave == 'Sparse Enclave' else 0)
        predicted_prob = np.clip(propensity_score, 0.1, 0.9)
        ipw_weight = actual_enclave / predicted_prob + (1 - actual_enclave) / (1 - predicted_prob)
        ipw_weight = np.clip(ipw_weight, 0.2, 5)  # Cap weights
        
        data.append({
            'id': i,
            'race': race,
            'cancer': cancer,
            'enclave': enclave,
            'age': age,
            'stage': stage,
            'ses': ses,
            'insurance': insurance,
            'survival_time': survival_time,
            'censored': censored,
            'event': not censored,
            'ipw_weight': ipw_weight
        })
    
    return pd.DataFrame(data)

def calculate_weighted_kaplan_meier(data, time_col='survival_time', event_col='event', 
                                  group_col='enclave', weight_col='ipw_weight'):
    """
    Calculate Kaplan-Meier estimates with IPW weights and confidence intervals.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Survival data
    time_col : str
        Column name for survival times
    event_col : str
        Column name for event indicator
    group_col : str
        Column name for grouping variable
    weight_col : str
        Column name for IPW weights
        
    Returns:
    --------
    dict
        Dictionary containing KM estimates for each group
    """
    results = {}
    
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group].copy()
        
        # Fit Kaplan-Meier with weights (approximation using frequency weights)
        kmf = KaplanMeierFitter()
        
        # Create weighted dataset by replicating observations
        weighted_times = []
        weighted_events = []
        
        for _, row in group_data.iterrows():
            weight = int(round(row[weight_col] * 10))  # Scale weights
            weighted_times.extend([row[time_col]] * weight)
            weighted_events.extend([row[event_col]] * weight)
        
        kmf.fit(weighted_times, weighted_events, label=group)
        
        results[group] = {
            'kmf': kmf,
            'survival_function': kmf.survival_function_,
            'confidence_interval': kmf.confidence_interval_,
            'median_survival': kmf.median_survival_time_,
            'timeline': kmf.timeline
        }
    
    return results

def plot_survival_curves(km_results, title="Kaplan-Meier Survival Curves", 
                        figsize=(12, 8), confidence_alpha=0.18):
    """
    Plot survival curves with confidence intervals.
    
    Parameters:
    -----------
    km_results : dict
        Results from calculate_weighted_kaplan_meier
    title : str
        Plot title
    figsize : tuple
        Figure size
    confidence_alpha : float
        Transparency for confidence intervals
    """
    plt.figure(figsize=figsize)
    
    colors = {
        'Dense Enclave': '#e74c3c',
        'Sparse Enclave': '#f39c12', 
        'No Enclave': '#3498db'
    }
    
    for group, result in km_results.items():
        kmf = result['kmf']
        color = colors.get(group, '#333333')
        
        # Plot survival curve
        kmf.plot(color=color, linewidth=3, label=group)
        
        # Plot confidence interval
        timeline = result['timeline']
        lower = result['confidence_interval'].iloc[:, 0]
        upper = result['confidence_interval'].iloc[:, 1]
        
        plt.fill_between(timeline, lower, upper, 
                        color=color, alpha=confidence_alpha, step='post')
    
    plt.xlabel('Time (Months)', fontsize=14, fontweight='bold')
    plt.ylabel('Survival Probability', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.xlim(0, 60)
    plt.ylim(0, 1)
    
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    return plt.gcf()

def calculate_survival_statistics(data, km_results):
    """
    Calculate key survival statistics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original survival data
    km_results : dict
        Kaplan-Meier results
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics table
    """
    stats_list = []
    
    for group in data['enclave'].unique():
        group_data = data[data['enclave'] == group]
        km_result = km_results[group]
        
        # Calculate 5-year survival (at 60 months)
        survival_func = km_result['survival_function']
        if len(survival_func) > 0:
            five_year_survival = survival_func.iloc[-1, 0] * 100
        else:
            five_year_survival = np.nan
            
        stats_list.append({
            'Enclave_Type': group,
            'N_Patients': len(group_data),
            'Events': group_data['event'].sum(),
            'Median_Survival_Months': km_result['median_survival'],
            'Five_Year_Survival_Pct': five_year_survival,
            'Censoring_Rate_Pct': (1 - group_data['event'].mean()) * 100
        })
    
    return pd.DataFrame(stats_list)

def perform_logrank_test(data):
    """
    Perform pairwise log-rank tests between enclave groups.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Survival data
        
    Returns:
    --------
    pd.DataFrame
        Log-rank test results
    """
    enclave_groups = data['enclave'].unique()
    results = []
    
    for i, group1 in enumerate(enclave_groups):
        for group2 in enclave_groups[i+1:]:
            data1 = data[data['enclave'] == group1]
            data2 = data[data['enclave'] == group2]
            
            result = logrank_test(
                data1['survival_time'], data2['survival_time'],
                data1['event'], data2['event']
            )
            
            results.append({
                'Group_1': group1,
                'Group_2': group2,
                'Test_Statistic': result.test_statistic,
                'P_Value': result.p_value,
                'Significant': result.p_value < 0.05
            })
    
    return pd.DataFrame(results)

def main():
    """
    Main analysis function.
    """
    print("🏥 Ethnic Enclave Cancer Survival Analysis")
    print("=" * 50)
    
    # Generate data
    print("Generating survival data...")
    data = generate_survival_data(n=8000)
    
    # Overall analysis
    print(f"\nDataset Overview:")
    print(f"Total patients: {len(data):,}")
    print(f"Cancer types: {', '.join(data['cancer'].unique())}")
    print(f"Racial groups: {', '.join(data['race'].unique())}")
    print(f"Enclave types: {', '.join(data['enclave'].unique())}")
    
    # Calculate Kaplan-Meier estimates
    print("\nCalculating Kaplan-Meier estimates...")
    km_results = calculate_weighted_kaplan_meier(data)
    
    # Generate survival curves
    print("Generating survival curves...")
    fig = plot_survival_curves(km_results, 
                              title="Cancer Survival by Ethnic Enclave Residence (All Cancers)")
    plt.show()
    
    # Calculate statistics
    print("\nSurvival Statistics:")
    stats_df = calculate_survival_statistics(data, km_results)
    print(stats_df.to_string(index=False, float_format='%.1f'))
    
    # Perform log-rank tests
    print("\nLog-rank Test Results:")
    logrank_df = perform_logrank_test(data)
    print(logrank_df.to_string(index=False, float_format='%.4f'))
    
    # Subgroup analyses
    print("\n" + "="*50)
    print("SUBGROUP ANALYSES")
    print("="*50)
    
    # Analysis by race
    for race in ['black', 'hispanic', 'asian']:
        race_data = data[data['race'] == race]
        if len(race_data) > 100:  # Only analyze if sufficient sample size
            print(f"\n{race.upper()} PATIENTS:")
            race_km = calculate_weighted_kaplan_meier(race_data)
            race_stats = calculate_survival_statistics(race_data, race_km)
            print(race_stats[['Enclave_Type', 'N_Patients', 'Five_Year_Survival_Pct']].to_string(index=False))
            
            # Plot race-specific curves
            fig = plot_survival_curves(race_km, 
                                     title=f"Cancer Survival - {race.title()} Patients")
            plt.show()
    
    # Analysis by cancer type
    for cancer in ['breast', 'lung', 'colon']:
        cancer_data = data[data['cancer'] == cancer]
        if len(cancer_data) > 100:
            print(f"\n{cancer.upper()} CANCER:")
            cancer_km = calculate_weighted_kaplan_meier(cancer_data)
            cancer_stats = calculate_survival_statistics(cancer_data, cancer_km)
            print(cancer_stats[['Enclave_Type', 'N_Patients', 'Five_Year_Survival_Pct']].to_string(index=False))
    
    print("\n" + "="*50)
    print("Analysis completed successfully!")
    print("="*50)
    
    return data, km_results

if __name__ == "__main__":
    # Run the analysis
    data, results = main()
    
    # Save results
    print("\nSaving results...")
    data.to_csv('ethnic_enclave_survival_data.csv', index=False)
    print("Data saved to: ethnic_enclave_survival_data.csv")
    
    # Additional analysis examples
    print("\nAdditional analysis examples:")
    print("1. Filter by specific race: data[data['race'] == 'black']")
    print("2. Filter by cancer type: data[data['cancer'] == 'lung']")
    print("3. Analyze race-cancer combinations: data[(data['race'] == 'hispanic') & (data['cancer'] == 'colon')]")
