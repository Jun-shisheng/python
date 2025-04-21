import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# 读取数据
file_path = "data/shill_bidding.csv"
df = pd.read_csv(file_path)

# 查看数据基本信息
df.info(), df.head()

# 提取特征和标签
X = df.drop(columns=["记录ID", "拍卖ID", "类别"])  # 去除无关列
y = df["类别"]

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
print('PCA降维前的数据形状为',X_train_pca.shape)
print('PCA降维后的数据形状为',X_test_pca.shape)
