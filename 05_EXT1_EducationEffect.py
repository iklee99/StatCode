import pandas as pd
import numpy as np
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
        stat, p_value = levene(*[data[data[group_col] == group][var].dropna() for group in data[group_col].unique()])
        results[var] = p_value
    return results

# Step 3: Regularity test
def regularity_test(data, dependent_vars, group_col):
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
    friedman_score = friedmanchisquare(*scores)
    friedman_satisfaction = friedmanchisquare(*satisfaction)
    return friedman_score, friedman_satisfaction

# Step 6: Post-hoc test
def post_hoc_test(data, dependent_var, group_col):
    mc = pairwise_tukeyhsd(data[dependent_var], data[group_col], alpha=0.05)
    return mc.summary()

# Main program to execute the analysis

data = load_data('datasets/education_methods_data.csv')
dependent_vars = ['Score', 'Satisfaction']
group_col = 'Education_Method'

var_test_results = variance_similarity_test(data, dependent_vars, group_col)
reg_test_results = regularity_test(data, dependent_vars, group_col)

if all(p > 0.05 for p in var_test_results.values()) and all(p > 0.05 for p in reg_test_results.values()):
    # Perform MANOVA
    manova_results = perform_manova(data, dependent_vars, group_col)
    print("MANOVA Results:\n", manova_results)
    for var in dependent_vars:
        print(f"\nPost-hoc test for {var}:")
        post_hoc_results = post_hoc_test(data, var, group_col)
        print(post_hoc_results)
else:
    # Perform Friedman test
    friedman_results = perform_friedman_test(data, dependent_vars, group_col)
    print("Friedman Test Results:\n", friedman_results)
    for var in dependent_vars:
        print(f"\nPost-hoc test for {var}:")
        post_hoc_results = post_hoc_test(data, var, group_col)
        print(post_hoc_results)
