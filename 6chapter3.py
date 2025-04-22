import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

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

# SVM分类器
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_pca, y_train)

# 预测测试集概率和类别
y_score = svm.predict_proba(X_test_pca)[:, 1]  # 获取正类的概率
y_pred = (y_score >= 0.5).astype(int)

# 绘制ROC曲线
def compute_roc(y_true, y_score, thresholds=np.linspace(0, 1, 100)):
    tpr_list = []
    fpr_list = []
    y_true = np.array(y_true)

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        TPR = TP / (TP + FN + 1e-10)
        FPR = FP / (FP + TN + 1e-10)
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return fpr_list, tpr_list

fpr, tpr = compute_roc(y_test, y_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="SVM ROC Curve", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("1-特异值")
plt.ylabel("灵敏度")
plt.title("ROC曲线")
plt.legend()
plt.grid(True)
plt.show()
