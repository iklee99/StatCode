import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
one_sample_df = pd.read_csv('datasets/t_test_one_sample.csv')
independent_samples_df = pd.read_csv('datasets/t_test_independent_samples.csv')
paired_samples_df = pd.read_csv('datasets/t_test_paired_samples.csv')

# One-sample t-test
population_mean = 50
one_sample_t_stat, one_sample_p_val = stats.ttest_1samp(one_sample_df['Sample'], population_mean)
print("One sample t-Test:")
print(f"t-stat={one_sample_t_stat}  p-value={one_sample_p_val}")

# Visualization for one-sample t-test
plt.figure(figsize=(10, 6))
sns.histplot(one_sample_df['Sample'], kde=True, color='skyblue', bins=10)
plt.axvline(x=population_mean, color='red', linestyle='--', label=f'Population Mean: {population_mean}')
plt.axvline(x=one_sample_df['Sample'].mean(), color='blue', linestyle='-', label=f'Sample Mean: {one_sample_df["Sample"].mean():.2f}')
plt.title('One-Sample t-Test')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Independent samples t-test
group1 = independent_samples_df[independent_samples_df['Group'] == 'Group1']['Value']
group2 = independent_samples_df[independent_samples_df['Group'] == 'Group2']['Value']
independent_t_stat, independent_p_val = stats.ttest_ind(group1, group2)
print("Independent Samples t-Test:")
print(f"t-stat={independent_t_stat}  p-value={independent_p_val}")

# Visualization for independent samples t-test
plt.figure(figsize=(10, 6))
sns.histplot(group1, kde=True, color='green', bins=10, label='Group1')
sns.histplot(group2, kde=True, color='orange', bins=10, label='Group2')
plt.axvline(x=group1.mean(), color='green', linestyle='-', label=f'Group1 Mean: {group1.mean():.2f}')
plt.axvline(x=group2.mean(), color='orange', linestyle='-', label=f'Group2 Mean: {group2.mean():.2f}')
plt.title('Independent Samples t-Test')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Paired samples t-test
paired_t_stat, paired_p_val = stats.ttest_rel(paired_samples_df['PreTest'], paired_samples_df['PostTest'])
print("Paired Samples t-Test:")
print(f"t-stat={paired_t_stat}  p-value={paired_p_val}")

# Visualization for paired samples t-test
plt.figure(figsize=(10, 6))
sns.histplot(paired_samples_df['PreTest'], kde=True, color='purple', bins=10, label='PreTest')
sns.histplot(paired_samples_df['PostTest'], kde=True, color='yellow', bins=10, label='PostTest')
plt.axvline(x=paired_samples_df['PreTest'].mean(), color='purple', linestyle='-', label=f'PreTest Mean: {paired_samples_df["PreTest"].mean():.2f}')
plt.axvline(x=paired_samples_df['PostTest'].mean(), color='yellow', linestyle='-', label=f'PostTest Mean: {paired_samples_df["PostTest"].mean():.2f}')
plt.title('Paired Samples t-Test')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

(one_sample_t_stat, one_sample_p_val), (independent_t_stat, independent_p_val), (paired_t_stat, paired_p_val)
