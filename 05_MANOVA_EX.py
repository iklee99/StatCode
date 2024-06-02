import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, shapiro, friedmanchisquare
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Step 1: Load CSV data
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Variance similarity test
def variance_similarity_test(data, dependent_vars, group_col):
    results = {}
    for var in dependent_vars:
        group_data = [data[data[group_col] == group][var].dropna() for group in data[group_col].unique()]
        group_data = [g for g in group_data if len(g) > 1]  # Ensure all groups have more than one sample
        if len(group_data) > 1:  # Ensure we have at least two groups to compare
            stat, p_value = levene(*group_data)
            results[var] = p_value
        else:
            results[var] = np.nan  # Set as NaN if not enough data
    return results

# Step 3: Normality test
def normality_test(data, dependent_vars):
    results = {}
    for var in dependent_vars:
        stat, p_value = shapiro(data[var])
        results[var] = p_value
    return results

# Step 4: MANOVA
def perform_manova(data, dependent_vars, group_col):
    formula = ' + '.join(dependent_vars) + ' ~ ' + group_col
    maov = MANOVA.from_formula(formula, data=data)
    return maov.mv_test()

# Step 5: Friedman test
def perform_friedman_test(data, dependent_vars, group_col):
    groups = data[group_col].unique()
    scores = [data[data[group_col] == group][dependent_vars[0]].values for group in groups]
    satisfaction = [data[data[group_col] == group][dependent_vars[1]].values for group in groups]

    # Ensure equal lengths for each group
    min_length = min(len(scores_group) for scores_group in scores)
    scores = [scores_group[:min_length] for scores_group in scores]
    satisfaction = [satisfaction_group[:min_length] for satisfaction_group in satisfaction]

    friedman_score = friedmanchisquare(*scores)
    friedman_satisfaction = friedmanchisquare(*satisfaction)
    return friedman_score, friedman_satisfaction

# Step 6: Post-hoc test
def post_hoc_test(data, dependent_var, group_col):
    mc = pairwise_tukeyhsd(data[dependent_var], data[group_col], alpha=0.05)
    return mc.summary()

# Step 7: Visualize the results
def visualize_results(data, dependent_vars, group_col):
    for var in dependent_vars:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_col, y=var, data=data)
        plt.title(f'Boxplot of {var} by {group_col}')
        plt.show()

# Main program to execute the analysis

data = load_data('datasets/medical_data_for_manova6.csv')
#data = load_data('datasets/medical_data_for_manova4.csv')
dependent_vars = ['Before', 'Middle', 'End']
group_col = 'Therapy'


# Ensure all groups have sufficient data
group_sizes = data[group_col].value_counts()
if all(size > 1 for size in group_sizes):
    var_test_results = variance_similarity_test(data, dependent_vars, group_col)
    norm_test_results = normality_test(data, dependent_vars)

    # Report which tests failed
    print("Variance Test Results:", var_test_results)
    print("Normality Test Results:", norm_test_results)
    failed_tests = {
        'variance': [var for var in var_test_results if var_test_results[var] <= 0.05 or np.isnan(var_test_results[var])],
        'normality': [var for var in norm_test_results if norm_test_results[var] <= 0.05]
    }
    print("Failed Variance Tests:", failed_tests['variance'])
    print("Failed Normality Tests:", failed_tests['normality'])

    # Determine if we should run MANOVA or Friedman test
    if len(failed_tests['variance']) == 0 and len(failed_tests['normality']) == 0:
        # Perform MANOVA
        try:
            manova_results = perform_manova(data, dependent_vars, group_col)
            print("MANOVA Results:\n", manova_results)
            for var in dependent_vars:
                print(f"\nPost-hoc test for {var}:")
                post_hoc_results = post_hoc_test(data, var, group_col)
                print(post_hoc_results)
        except Exception as e:
            print("Error performing MANOVA:", e)
    else:
        # Perform Friedman test
        friedman_results = perform_friedman_test(data, dependent_vars, group_col)
        print("Friedman Test Results:\n", friedman_results)
        for var in dependent_vars:
            print(f"\nPost-hoc test for {var}:")
            post_hoc_results = post_hoc_test(data, var, group_col)
            print(post_hoc_results)

    # Visualize the results
    visualize_results(data, dependent_vars, group_col)
else:
    print("Error: Not all groups have sufficient data for the analysis.")
