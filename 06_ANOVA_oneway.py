import pandas as pd
import numpy as np
from scipy.stats import f_oneway, levene, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the data from CSV file
data = pd.read_csv('datasets/anova_oneway_data.csv')

# Extract groups
group1 = data[data['group'] == 'Group1']['value']
group2 = data[data['group'] == 'Group2']['value']
group3 = data[data['group'] == 'Group3']['value']
group4 = data[data['group'] == 'Group4']['value']

# Homogeneity test using Levene's test
levene_stat, levene_p = levene(group1, group2, group3, group4)

# Normality test using Shapiro-Wilk test
shapiro_p_values = [shapiro(group)[1] for group in [group1, group2, group3, group4]]

# One-Way ANOVA
anova_results = f_oneway(group1, group2, group3, group4)
f_value, p_value = anova_results.statistic, anova_results.pvalue

# Bonferroni post-hoc test
model = ols('value ~ C(group)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
mc = pairwise_tukeyhsd(data['value'], data['group'], alpha=0.05)
bonferroni_results = mc.summary()

# Effect size: Eta Squared and Partial Eta Squared
sum_sq_between = anova_table['sum_sq'][0]
sum_sq_total = sum_sq_between + anova_table['sum_sq'][1]
eta_squared = sum_sq_between / sum_sq_total
partial_eta_squared = sum_sq_between / (sum_sq_between + anova_table['sum_sq'][1])

# Effect size: Cohen's f
cohen_f = np.sqrt(eta_squared / (1 - eta_squared))

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for groups
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='value', data=data)
plt.title('Boxplot of Groups')
plt.savefig('figures/ANOVA_oneway/group_boxplot.png')

# ANOVA table
anova_table

# Pairwise comparison
#bonferroni_results

#import ace_tools as tools; tools.display_dataframe_to_user(name="ANOVA Data", dataframe=data)

#levene_stat, levene_p, shapiro_p_values, f_value, p_value, bonferroni_results

# Display results
results = {
    "Levene's Test Statistic": levene_stat,
    "Levene's Test p-value": levene_p,
    "Shapiro-Wilk p-values": shapiro_p_values,
    "ANOVA F-value": f_value,
    "ANOVA p-value": p_value,
    "Bonferroni Post-Hoc Test Results": bonferroni_results
}

results

print("\n")
print(f"levene_stat = {levene_stat}  levene_p = {levene_p}\n")
print(f"shapiro_p_values = {shapiro_p_values}\n")
print(f"anova_result = {anova_results}\n")
print(f"Effect size (eta_squared) = {eta_squared}")
print(f"Effect size (partial_eta_squared) = {partial_eta_squared}")
print(f"Effect size (cohen_f) = {cohen_f}\n")
print(f"bonferroni_result\n {bonferroni_results}")
