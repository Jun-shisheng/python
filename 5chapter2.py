import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 读取数据
df = pd.read_csv('data/aqi.csv', parse_dates=['日期'])

# 筛选2023年1-9月的数据
df = df[(df['日期'].dt.year == 2023) & (df['日期'].dt.month <= 9)].copy()

# 质量等级分类散点图
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='日期',
                y='AQI', hue='质量等级',
                palette={'优':'green', '良':'blue', '轻度污染':'orange', '中度污染':'red', '重度污染':'purple'},
                s=60)
plt.title('2023年1-9月空气质量等级分布', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('AQI指数', fontsize=12)
plt.legend(title='质量等级', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()

# PM2.5与AQI线性回归拟合图
plt.figure(figsize=(10, 6))

# 获取数据
X = df['PM2.5含量/ppm'].values
y = df['AQI'].values

# 用 NumPy 进行线性拟合
slope, intercept = np.polyfit(X, y, 1)  # 线性拟合 y = slope * x + intercept
y_pred = slope * X + intercept

# 计算R²值（决定系数）
y_mean = np.mean(y)
ss_total = np.sum((y - y_mean) ** 2)  # 总方差
ss_residual = np.sum((y - y_pred) ** 2)  # 残差平方和
r2 = 1 - (ss_residual / ss_total)  # R²计算

# 计算相关系数
correlation = np.corrcoef(X, y)[0, 1]

# 绘制散点图和拟合直线
sns.scatterplot(x=X, y=y, color='blue', alpha=0.6)
plt.plot(X, y_pred, color='red', linewidth=2,
         label=f'线性回归: y = {slope:.2f}x + {intercept:.2f}')

# 添加统计信息
plt.text(0.05, 0.95, f'R² = {r2:.3f}\n相关系数 = {correlation:.3f}',
         transform=plt.gca().transAxes, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.8))

plt.title('PM2.5浓度与AQI线性关系', fontsize=14)
plt.xlabel('PM2.5浓度(ppm)', fontsize=12)
plt.ylabel('AQI指数', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 空气质量相关性热力图
# 选择分析的指标
corr_columns = ['AQI', 'PM2.5含量/ppm', 'PM10含量/ppm', 'SO2含量/ppm', 'CO含量/ppm', 'NO2含量/ppm', 'O3含量/ppm']
corr_df = df[corr_columns]

# 计算相关系数矩阵
corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            center=0, linewidths=0.5, annot_kws={"size": 12})

plt.title('空气质量指标相关性热力图', fontsize=14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10, rotation=0)
plt.tight_layout()
plt.show()

# 输出关键统计数据
print("\n=== 关键统计数据 ===")
print(f"1. AQI均值: {df['AQI'].mean():.1f}")
print(f"2. PM2.5均值: {df['PM2.5含量/ppm'].mean():.1f} ppm")
print(f"3. 质量等级分布:\n{df['质量等级'].value_counts()}")
print("\n4. 各污染物与AQI的相关系数:")
print(corr_matrix['AQI'].sort_values(ascending=False))
