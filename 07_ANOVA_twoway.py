import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('datasets/anova_twoway_data.csv')  # Replace with your actual file path

# 2-1) Homogeneity test using Levene's test
levene_stat, levene_p = stats.levene(data['Value'][data['Factor_A'] == 'A1'],
                                     data['Value'][data['Factor_A'] == 'A2'])
print(f"Levene's test for homogeneity: stat={levene_stat}, p-value={levene_p}")

# 2-2) Normality test using Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(data['Value'])
print(f"Shapiro-Wilk test for normality: stat={shapiro_stat}, p-value={shapiro_p}")

# 2-3) Two-way ANOVA
model = ols('Value ~ C(Factor_A) * C(Factor_B)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("Two-way ANOVA results:")
print(anova_table)

# 2-4) Bonferroni post-hoc tests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

m_comp = pairwise_tukeyhsd(endog=data['Value'], groups=data['Factor_A'] + data['Factor_B'], alpha=0.05)
print("Bonferroni post-hoc test results:")
print(m_comp)

# 2-5) Visualization
# Boxplot for the factors
plt.figure(figsize=(10, 6))
sns.boxplot(x='Factor_A', y='Value', hue='Factor_B', data=data)
plt.title('Boxplot of Factor A and Factor B interaction')
plt.show()

# Interaction plot
plt.figure(figsize=(10, 6))
sns.pointplot(x='Factor_A', y='Value', hue='Factor_B', data=data, dodge=True, markers=['o', 's', 'D'], linestyles=['-', '--', ':'])
plt.title('Interaction plot of Factor A and Factor B')
plt.show()
