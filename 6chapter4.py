import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
print("降维后的训练集形状:", X_train_pca.shape)
print("降维后的测试集形状:", X_test_pca.shape)

# 构建线性回归模型
linear_reg = LinearRegression()
linear_reg.fit(X_train_pca, y_train)

# 预测测试集结果
y_pred = linear_reg.predict(X_test_pca)

# 计算模型的评价指标
mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # R方值

# 输出模型性能评价
print("\n线性回归模型评价：")
print(f"平均绝对误差 (MAE): {mae}")
print(f"均方误差 (MSE): {mse}")
print(f"R方值 (R^2): {r2}")

# 根据R方值评判模型性能
if r2 > 0.8:
    print("模型性能良好")
elif r2 > 0.5:
    print("模型表现一般")
else:
    print("模型性能较差")