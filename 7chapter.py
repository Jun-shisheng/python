import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 只保留这行即可
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 自定义函数：用户名首字母转ASCII
def to_code(name):
    if isinstance(name, str) and len(name) > 0:
        first_char = name[0].upper()
        return ord(first_char) if first_char.isalpha() else 0
    return 0

# 自定义函数：在线时长分段（小时）
def duration_segment(duration):
    if pd.isnull(duration):
        return '未知'
    if duration < 1:
        return '0-1小时'
    elif duration < 3:
        return '1-3小时'
    elif duration < 5:
        return '3-5小时'
    else:
        return '5小时以上'

# 读取CSV文件
df = pd.read_csv('data/某APP用户信息数据.csv')

# 缺失值填充
df['不愿分享概率'] = df['不愿分享概率'].fillna(0.0)
df['愿意分享概率'] = df['愿意分享概率'].fillna(0.0)

# 限制概率在[0, 1]之间
df['不愿分享概率'] = df['不愿分享概率'].apply(lambda x: 0 if x < 0 else (1 if x > 1 else x))

# 是否点击分享映射为数值
df['是否点击分享'] = df['是否点击分享'].map({'T': 1, 'F': 0})

# 用户名编码
df['用户名编码'] = df['用户名'].apply(to_code)

# 在线时长分段（将分钟转换为小时）
df['在线时长分段'] = (df['在线时长/分钟'] / 60).apply(duration_segment)

# 显示前几行结果
print(df.head())

# 可选保存
# df.to_csv('处理后的用户信息数据.csv', index=False)

# 构建特征集（数值特征）
X = df[['不愿分享概率', '愿意分享概率', '用户名编码', '在线时长/分钟']]

# 标签
y_true = df['是否点击分享']

# 构建 KMeans 模型（聚类数 = 2）
kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_predict(X)

# 使用 PCA 将数据降至 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化聚类结果（KMeans 预测）
plt.figure(figsize=(10, 4))

# 子图1：聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=30)
plt.title('KMeans 聚类结果')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# 子图2：真实标签
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='coolwarm', s=30)
plt.title('真实标签（是否点击分享）')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

plt.tight_layout()
plt.show()

# 使用 FMI 指标进行评价
fmi_score = fowlkes_mallows_score(y_true, y_pred)
print("Fowlkes-Mallows Index (FMI):", round(fmi_score, 4))

# 根据FMI得分做结论
if fmi_score > 0.8:
    print("结论：聚类结果与真实标签高度匹配，模型效果较好。")
elif fmi_score > 0.6:
    print("结论：聚类结果与真实标签有一定匹配，模型效果中等。")
else:
    print("结论：聚类结果与真实标签匹配度较低，模型效果不佳。")
