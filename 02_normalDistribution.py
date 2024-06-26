import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# 1. 데이터셋 읽기
file_path = 'datasets/normal30.csv'  # 29개 데이터
#file_path = 'datasets/normal100.csv'  # 100개 데이터
data = pd.read_csv(file_path)

# 데이터가 단일 열에 있다고 가정
sample_data = data['value'].dropna()

# 2. 분포 분석

# 정규분포 (Normal Distribution)
mu, std = norm.fit(sample_data)

# Z-분포 (Z-Distribution)
z_scores = (sample_data - mu) / std

# 주어진 Z score 값에 대한 확률 계산 (Standard Normal Table)
z_value = 1.96
probability = norm.cdf(z_value)
print(f"Z score 1.96 의 확률 (Standard Normal Table) = {probability}")

# 주어진 확률에 대한 Z score 계산
probability = 0.975
z_score = norm.ppf(probability)
print(f"확률 0.975에 대한 Z-score = {z_score}")
probability = 0.995
z_score = norm.ppf(probability)
print(f"확률 0.995에 대한 Z-score = {z_score}")

# T-분포 (T-Distribution)
degrees_of_freedom = len(sample_data) - 1
t_scores = (sample_data - np.mean(sample_data)) / (np.std(sample_data, ddof=1) / np.sqrt(len(sample_data)))

# 3. 플로팅

# 정규분포 (Normal Distribution)
x = np.linspace(min(sample_data), max(sample_data), 1000)
plt.figure()
plt.plot(x, norm.pdf(x, mu, std), label='Normal Distribution')
plt.hist(sample_data, bins=30, density=True, alpha=0.6, color='g', label='Sample Data')
plt.legend()
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Z-분포 (Z-Distribution)
x = np.linspace(min(z_scores), max(z_scores), 1000)
plt.figure()
plt.plot(x, norm.pdf(x, 0, 1), label='Z-Distribution')
plt.hist(z_scores, bins=30, density=True, alpha=0.6, color='g', label='Sample Data (Z-Scores)')
plt.legend()
plt.title('Z Distribution')
plt.xlabel('Z-Score')
plt.ylabel('Density')
plt.show()

# T-분포 (T-Distribution)
x = np.linspace(min(t_scores), max(t_scores), 1000)
plt.figure()
plt.plot(x, t.pdf(x, degrees_of_freedom), label=f'T-Distribution (df={degrees_of_freedom})')
plt.hist(t_scores, bins=30, density=True, alpha=0.6, color='g', label='Sample Data (T-Scores)')
plt.legend()
plt.title('T Distribution')
plt.xlabel('T-Score')
plt.ylabel('Density')
plt.show()
