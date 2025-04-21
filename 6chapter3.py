import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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

# 使用StandardScaler进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用sklearn的PCA进行降维
pca = PCA(n_components=0.999)  # 保留99.9%的方差
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 输出降维后的数据形状
print("降维后的训练集形状:", X_train_pca.shape)
print("降维后的测试集形状:", X_test_pca.shape)

# 构建支持向量机分类模型
svm = SVC(random_state=42)
svm.fit(X_train_pca, y_train)

# 预测测试集前10个数据的结果
y_pred = svm.predict(X_test_pca[:10])

# 打印预测结果
print("\n测试集前10个数据的预测结果:", y_pred)

# 输出分类报告
print("\n分类模型评价报告:\n", classification_report(y_test, svm.predict(X_test_pca), target_names=["0", "1"]))