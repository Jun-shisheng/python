import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 手动指定列名（假设共有4列，服装尺寸在第4列）
column_names = ['体重/kg', '年龄/岁', '身高/cm', '服装尺寸']

# 读取CSV文件，无表头
df = pd.read_csv('data/size_data.csv', header=None, names=column_names)

# 去除“服装尺寸”中多余空格、统一大小写
df['服装尺寸'] = df['服装尺寸'].astype(str).str.strip().str.upper()

# 只转换数值列为数值型，非数字设为 NaN
df[['体重/kg', '年龄/岁', '身高/cm']] = df[['体重/kg', '年龄/岁', '身高/cm']].apply(pd.to_numeric, errors='coerce')

# 删除数值列或服装尺寸中存在缺失值的行
df_cleaned = df.dropna(subset=['体重/kg', '年龄/岁', '身高/cm', '服装尺寸'])

# 删除异常值：年龄小于18岁或体重小于30kg
df_cleaned = df_cleaned[(df_cleaned['年龄/岁'] >= 18) & (df_cleaned['体重/kg'] >= 30)]

# 检查清洗后是否还有数据
if df_cleaned.shape[0] == 0:
    raise ValueError("清洗后数据集为空，请检查原始数据是否有有效记录。")

# 检查缺失值
print("\n缺失值检查:")
print(df_cleaned.isnull().sum())

# 打印数据范围
print("\n年龄范围:", (df_cleaned['年龄/岁'].min(), df_cleaned['年龄/岁'].max()))
print("体重范围:", (df_cleaned['体重/kg'].min(), df_cleaned['体重/kg'].max()))
print("身高范围:", (df_cleaned['身高/cm'].min(), df_cleaned['身高/cm'].max()))

# 计算BMI
df_cleaned['BMI'] = df_cleaned['体重/kg'] / ((df_cleaned['身高/cm'] / 100) ** 2)

# 构建 BMI_range 特征
def classify_bmi(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 24:
        return 1
    elif bmi < 28:
        return 2
    else:
        return 3

df_cleaned['BMI_range'] = df_cleaned['BMI'].apply(classify_bmi)

# 显示样例数据
print("\n处理后的数据样例:")
print(df_cleaned[['年龄/岁', '身高/cm', '体重/kg', '服装尺寸', 'BMI', 'BMI_range']].head())

# SVM建模
features = ['年龄/岁', '身高/cm', '体重/kg', 'BMI', 'BMI_range']
target = '服装尺寸'

X = df_cleaned[features]
y = df_cleaned[target]

# 再次确认是否足够数据进行训练
if len(X) < 2:
    raise ValueError("有效样本太少，无法进行训练。请检查数据清洗逻辑或源数据内容。")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# 预测与评估
y_pred = svm_model.predict(X_test)

print("\n模型评估:")
print("准确率:", accuracy_score(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
