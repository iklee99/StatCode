import numpy as np
from statsmodels.stats.power import (
    TTestPower, TTestIndPower, GofChisquarePower, NormalIndPower, FTestAnovaPower
)
from statsmodels.stats.proportion import proportion_effectsize

# One-Sample t-Test
one_sample_ttest = TTestPower()
sample_size_one_sample = one_sample_ttest.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"One-sample t-test: 최소 표본 크기 = {sample_size_one_sample:.2f}")
print(f"with effect_size=0.5, alpha=0.05, power=0.8")
print("\n")

# Independent Samples t-Test
independent_ttest = TTestIndPower()
sample_size_independent = independent_ttest.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Independent samples t-test: 최소 표본 크기 (각 그룹 당) = {sample_size_independent:.2f}")
print(f"with effect_size=0.5, alpha=0.05, power=0.8")
print("\n")

# Paired t-Test
paired_ttest = TTestPower()
sample_size_paired = paired_ttest.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Paired t-test: 최소 표본 크기 = {sample_size_paired:.2f}")
print(f"with effect_size=0.5, alpha=0.05, power=0.8")
print("\n")

# Binomial Test
effect_size_binomial = proportion_effectsize(0.5, 0.6)  # 두 비율 간의 차이
binomial_test = NormalIndPower()
sample_size_binomial = binomial_test.solve_power(effect_size=effect_size_binomial, alpha=0.05, power=0.8)
print(f"Binomial test: 최소 표본 크기 = {sample_size_binomial:.2f}")
print(f"with effect_size_binomial={effect_size_binomial} alpha=0.05 power=0.8")
print("\n")

# Chi-Square Test
chi_square_test = GofChisquarePower()
sample_size_chi_square = chi_square_test.solve_power(effect_size=0.3, alpha=0.05, power=0.8, n_bins=5)
print(f"Chi-square test: 최소 표본 크기 = {sample_size_chi_square:.2f}")
print(f"with effect_size=0.3, alpha=0.05, power=0.8, n_bins=5")
print("\n")

# One-Way ANOVA
one_way_anova = FTestAnovaPower()
sample_size_one_way_anova = one_way_anova.solve_power(effect_size=0.25, alpha=0.05, power=0.8, k_groups=3)
print(f"One-way ANOVA: 최소 표본 크기 (각 그룹 당) = {sample_size_one_way_anova:.2f}")
print(f"with effect_size=0.25, alpha=0.05, power=0.8, k_groups=3")
print("\n")

# Repeated Measures ANOVA
# 참고: 여기서는 반복 측정에 대한 직접적인 파워 분석 라이브러리가 없으므로, 통상적인 일원 분산분석과 비슷한 방식으로 접근
# 실제 연구에서는 G*Power 같은 소프트웨어를 이용하거나, 시뮬레이션 기반 접근이 필요할 수 있음
effect_size = 0.25
sample_size_repeated_anova = one_way_anova.solve_power(effect_size=0.25, alpha=0.05, power=0.8, k_groups=3)
print(f"Repeated Measures ANOVA: 최소 표본 크기 (각 그룹 당) = {sample_size_repeated_anova:.2f}")
print(f"with effect_size=0.25, alpha=0.05, power=0.8, k_groups=3")
print("\n")

# Two-Way ANOVA
# 참고: Statsmodels는 반복 측정을 직접 지원하지 않으므로, 표준 ANOVA의 두 개의 주 효과와 상호작용 항을 포함한 모형 사용
two_way_anova = FTestAnovaPower()
sample_size_two_way_anova = two_way_anova.solve_power(effect_size=0.25, alpha=0.05, power=0.8, k_groups=4)  # 예: 두 요인 각각 2수준
print(f"Two-way ANOVA: 최소 표본 크기 (각 그룹 당) = {sample_size_two_way_anova:.2f}")
print(f"with effect_size={0.25} alpha={0.05} power={0.8} k_groups={4}")
print("\n")

# Two-Way ANOVA with Repeated Measures
# 참고: 여기서는 반복 측정에 대한 직접적인 파워 분석 라이브러리가 없으므로, 통상적인 ANOVA와 비슷한 방식으로 접근
sample_size_two_way_repeated_anova = two_way_anova.solve_power(effect_size=0.25, alpha=0.05, power=0.8, k_groups=4)
print(f"Two-way ANOVA with repeated measures: 최소 표본 크기 (각 그룹 당) = {sample_size_two_way_repeated_anova:.2f}")
print(f"with effect_size={0.25} alpha={0.05} power={0.8} k_groups={4}")
print("\n")

# Levene's Test
# Levene's Test는 표본 크기 계산을 위한 전통적인 파워 분석 방법이 부족함
# 보통 경험적으로 최소 20-30개의 샘플을 권장

# Mann-Whitney U Test
mannwhitneyu = TTestIndPower()  # 유사한 방법으로 접근 가능
sample_size_mannwhitneyu = mannwhitneyu.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Mann-Whitney U test: 최소 표본 크기 (각 그룹 당) = {sample_size_mannwhitneyu:.2f}")
print(f"with effect_size={0.5} alpha={0.05} power={0.8}")
print("\n")

# Wilcoxon Test
wilcoxon = TTestPower()  # 유사한 방법으로 접근 가능
sample_size_wilcoxon = wilcoxon.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Wilcoxon test: 최소 표본 크기 = {sample_size_wilcoxon:.2f}")
print(f"with effect_size={0.5} alpha={0.05} power={0.8}")
print("\n")

# Kruskal-Wallis Test
kruskalwallis = FTestAnovaPower()
effect_size = 0.25
k_groups = 3
sample_size_kruskalwallis = kruskalwallis.solve_power(effect_size=0.25, alpha=0.05, power=0.8, k_groups=3)
print(f"Kruskal-Wallis test: 최소 표본 크기 (각 그룹 당) = {sample_size_kruskalwallis:.2f}")
print(f"with effect_size={0.25} alpha={0.05} power={0.8} k_groups={3}")
print("\n")

# Friedman Test
# 참고: Friedman test는 비모수 검정으로, 통상적인 파워 분석 방법이 부족함
# 보통 경험적으로 최소 10-20개의 샘플을 권장

# Pearson Correlation
from statsmodels.stats.power import NormalIndPower
effect_size_pearson = 0.3  # 상관계수 r
sample_size_pearson = NormalIndPower().solve_power(effect_size=effect_size_pearson, alpha=0.05, power=0.8)
print(f"Pearson correlation: 최소 표본 크기 = {sample_size_pearson:.2f}")
print(f"with effect_size_pearson={effect_size_pearson} alpha=0.05 power=0.8")
print("\n")

# Spearman Correlation
sample_size_spearman = sample_size_pearson  # Pearson과 유사하게 접근
print(f"Spearman correlation: 최소 표본 크기 = {sample_size_spearman:.2f}")
print(f"with effect_size_pearson={effect_size_pearson} alpha=0.05 power=0.8")
print("\n")

# Point-Biserial Correlation
sample_size_pointbiserial = sample_size_pearson  # 유사하게 접근
print(f"Point-biserial correlation: 최소 표본 크기 = {sample_size_pointbiserial:.2f}")
print(f"with effect_size_pearson={effect_size_pearson} alpha=0.05 power=0.8")
print("\n")

# Partial correlation
# 참고: Partial correlation의 경우, 표본 크기 계산을 위한 전통적인 파워 분석 방법이 부족함
# 보통 경험적으로 최소 30개의 샘플을 권장

# Linear Regression
from statsmodels.stats.power import tt_solve_power
sample_size_linear_regression = tt_solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Linear regression: 최소 표본 크기 = {sample_size_linear_regression:.2f}")
print("with effect_size=0.5 alpha=0.05 power=0.8")
print("\n")

# Logistic Regression
sample_size_logistic_regression = tt_solve_power(effect_size=0.5, alpha=0.05, power=0.80)
print(f"Logistic regression: 최소 표본 크기 = {sample_size_logistic_regression:.2f}")
print("with effect_size=0.5 alpha=0.05 power=0.8")
print("\n")



