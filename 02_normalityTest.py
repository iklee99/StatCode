import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = 'datasets/normality.csv'  # CSV 파일 경로를 입력하세요
data = pd.read_csv(file_path)

# 특정 열을 선택하여 데이터 추출 (예시에서는 'value' 열을 사용)
data_column = data['value']

# Kolmogorov-Smirnov Test
ks_statistic, ks_p_value = stats.kstest(data_column, 'norm')
print(f"Kolmogorov-Smirnov Test: Statistic={ks_statistic}, p-value={ks_p_value}")

print("H0: Dataset은 normally distributed 되어 있다")
if ks_p_value > 0.05:
    print("p-value > 0.05 이므로 H0를 reject 할 수 없다")
else:
    print("p-value <= 0.05 이므로 H0를 reject 한다")
    print("즉, Dataset은 normally distributed 되어 있지 않다")
print("\n")
          

# Shapiro-Wilk Test
sw_statistic, sw_p_value = stats.shapiro(data_column)
print(f"Shapiro-Wilk Test: Statistic={sw_statistic}, p-value={sw_p_value}")

print("H0: Dataset은 normally distributed 되어 있다")
if sw_p_value > 0.05:
    print("p-value > 0.05 이므로 H0를 reject 할 수 없다")
else:
    print("p-value <= 0.05 이므로 H0를 reject 한다")
    print("즉, Dataset은 normally distributed 되어 있지 않다")
print("\n")

# Anderson-Darling Test
ad_result = stats.anderson(data_column, dist='norm')
print(f"Anderson-Darling Test: Statistic={ad_result.statistic}")
isNormal = True

for i in range(len(ad_result.critical_values)):
    sl, cv = ad_result.significance_level[i], ad_result.critical_values[i]
    print(f"Significance Level: {sl}, Critical Value: {cv}")
    if (ad_result.statistic < ad_result.critical_values[i]):
        print(f"statistics value {ad_result.statistic} < critical value {ad_result.critical_values[i]}")
    else: 
        print(f"statistics value {ad_result.statistic} >= critical value {ad_result.critical_values[i]}")
        isNormal = False
if (isNormal == True):
    print("모든 Significance Level에서 statistics value < critical value 이므로 Dataset은 normally distributed")
else:
    print("모든 Significance Level에서 statistics value < critical value 가 아니므로 Dataset은 normally distributed 되어 있지 않다")

# Q-Q Plot
fig, ax = plt.subplots()
stats.probplot(data_column, dist="norm", plot=ax)
ax.get_lines()[1].set_color('r')
plt.show()
