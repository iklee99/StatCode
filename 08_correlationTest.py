import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, pearsonr, spearmanr

# Load CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Normality test using Shapiro-Wilk test
def normality_test(data):
    stat, p = shapiro(data)
    return stat, p

# Correlation test decision based on normality
def correlation_test(data1, data2):
    stat_shapiro_x, p_shapiro_x = normality_test(data1)
    stat_shapiro_y, p_shapiro_y = normality_test(data2)
    
    results = {
        'shapiro_x': {'stat': stat_shapiro_x, 'p': p_shapiro_x},
        'shapiro_y': {'stat': stat_shapiro_y, 'p': p_shapiro_y}
    }
    
    if p_shapiro_x > 0.05 and p_shapiro_y > 0.05:
        corr_stat, corr_p = pearsonr(data1, data2)
        method = 'Pearson'
    else:
        corr_stat, corr_p = spearmanr(data1, data2)
        method = 'Spearman'
    
    results['correlation'] = {'method': method, 'stat': corr_stat, 'p': corr_p}
    return results

# Visualization
def visualize(data1, data2, results):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.scatterplot(x=data1, y=data2, ax=axs[0])
    axs[0].set_title('Scatter Plot')
    
    sns.histplot(data1, kde=True, ax=axs[1], color='blue', label='x')
    sns.histplot(data2, kde=True, ax=axs[1], color='red', label='y')
    axs[1].legend()
    axs[1].set_title('Distribution of x and y')
    
    plt.show()
    
    print("Shapiro Test for x: stat = {:.4f}, p = {:.4f}".format(results['shapiro_x']['stat'], results['shapiro_x']['p']))
    print("Shapiro Test for y: stat = {:.4f}, p = {:.4f}".format(results['shapiro_y']['stat'], results['shapiro_y']['p']))
    print("{} Correlation Test: stat = {:.4f}, p = {:.4f}".format(results['correlation']['method'], results['correlation']['stat'], results['correlation']['p']))

# Example usage
file_path = 'datasets/correlation_test_data.csv'  # replace with your file path
df = load_data(file_path)
data1 = df['x']
data2 = df['y']
results = correlation_test(data1, data2)
visualize(data1, data2, results)
