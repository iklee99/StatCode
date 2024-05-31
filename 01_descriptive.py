import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# CSV 파일 로드
data = pd.read_csv('./datasets/heart.csv')


# 기술통계 계산
print("기술통계 요약:")
print(data.describe())

print("\n평균:")
print(data.mean())

print("\n중앙값:")
print(data.median())

print("\n최빈값:")
print(data.mode().iloc[0])

print("\n범위:")
print(data.max() - data.min())

print("\nIQR (Interquartile Range):")
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print("\n표준편차:")
print(data.std())

print("\n분산:")
print(data.var())

print("\n왜도:")
print(data.apply(skew))

print("\n첨도:")
print(data.apply(kurtosis))

# 히스토그램
data.hist(bins=15, figsize=(15, 6), layout=(2, 3))
plt.tight_layout()
plt.show()

# 상자 그림
# box plot options: https://jimmy-ai.tistory.com/51
plt.figure(figsize=(10, 4))
sns.boxplot(data=data, whis=1000.0)
plt.title('Box Plots of All Variables')
plt.show()

# 산점도 행렬
sns.pairplot(data)
plt.suptitle('Scatter Plots of All Variables', y=1.02)
plt.show()
