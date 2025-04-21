import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, v_measure_score, fowlkes_mallows_score
import os

# 禁用调试器的文件验证部分
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# 尝试读取数据
try:
    file_path = "data/shill_bidding.csv"
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

# 查看数据基本信息
print(df.info())
print(df.head())

# 提取特征和标签
X = df[["竞标者倾向", "竞标比率", "连续竞标"]]
y = df["类别"]

# 将类别标签转换为0和1，假设0是正常，1是不正常
y = y.map({0: 0, 1: 1})

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用MinMaxScaler进行标准化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用sklearn的PCA进行降维
pca = PCA(n_components=0.999)  # 保留99.9%的方差
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 输出降维后的数据形状
print("Training set after PCA:", X_train_pca.shape)
print("Test set after PCA:", X_test_pca.shape)

# 聚类：KMeans, 使用聚类数目为 2 来评估是否能够分为两类
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_pca)

# 预测训练集的聚类标签
y_train_pred = kmeans.predict(X_train_pca)

# 计算各项评价指标
ari = adjusted_rand_score(y_train, y_train_pred)
v_measure = v_measure_score(y_train, y_train_pred)
fmi = fowlkes_mallows_score(y_train, y_train_pred)

# 输出聚类评价指标
print(f"\nFor n_clusters = 2:")
print(f"ARI评价指标: {ari}")
print(f"V-Measure评价指标: {v_measure}")
print(f"FMI评价指标: {fmi}")
