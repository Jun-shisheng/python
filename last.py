import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import os

# 设置全局样式
sns.set_theme(style="whitegrid")  # 使用seaborn的白色网格主题
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 1. 数据加载与预处理
# 设置数据路径
DATA_DIR = "data"
FILE_NAME = "open-meteo-52.55N13.41E38m.csv"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# 读取数据（跳过元数据行）
def load_and_preprocess(filepath):
    """加载并预处理数据"""
    try:
        df = pd.read_csv(filepath, skiprows=3)
    except FileNotFoundError:
        raise FileNotFoundError(f"无法找到数据文件: {filepath}. 请确保文件路径正确。")

    # 时间处理 - 使用更灵活的方式解析时间
    try:
        # 首先尝试ISO8601格式
        df['time'] = pd.to_datetime(df['time'], format='ISO8601')
    except ValueError:
        try:
            # 如果失败，尝试混合格式
            df['time'] = pd.to_datetime(df['time'], format='mixed')
        except ValueError:
            # 最后尝试强制转换，无法解析的设为NaT
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # 检查是否有无法解析的时间
    if df['time'].isnull().any():
        print(f"警告: 发现 {df['time'].isnull().sum()} 个无法解析的时间值，将被删除")
        df = df.dropna(subset=['time'])

    df.set_index('time', inplace=True)

    # 列名清洗
    df.columns = [col.split(' (')[0].strip().lower().replace(' ', '_') for col in df.columns]

    # 缺失值处理
    print("\n缺失值统计：")
    print(df.isnull().sum())

    # 使用中位数填充缺失值
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df),
                              columns=df.columns,
                              index=df.index)

    return df_imputed

try:
    df = load_and_preprocess(FILE_PATH)
except Exception as e:
    print(f"数据加载错误: {e}")
    exit()

# 2. 探索性数据分析 (EDA)
def plot_combined_trends(data):
    """合并温度、湿度、降水趋势图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 温度
    data['temperature_2m'].plot(ax=axes[0], color='tomato')
    axes[0].set_title("温度趋势图", pad=10)

    # 湿度
    data['relative_humidity_2m'].plot(ax=axes[1], color='steelblue')
    axes[1].set_title("相对湿度趋势图", pad=10)

    # 降水
    data['precipitation'].plot(ax=axes[2], color='dodgerblue', alpha=0.5)
    axes[2].set_title("降水趋势图", pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'combined_trends.png'))
    plt.show()


def plot_correlation_heatmap(data):
    corr = data.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", square=True,
        cmap='coolwarm', center=0, cbar_kws={"shrink":0.75}
    )
    plt.title("Feature Correlation Matrix", pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'correlation_heatmap.png'))
    plt.show()


# 执行EDA可视化
plot_combined_trends(df)
plot_correlation_heatmap(df)


# 3. PCA降维与KMeans聚类
def perform_pca_and_clustering(data, n_clusters=3):
    """执行PCA降维和聚类分析"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # PCA降维（保留95%方差）
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # 可视化方差解释率
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_,
            color='steelblue', alpha=0.7)
    plt.plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_),
             color='tomato', marker='o')
    plt.title("PCA主成分方差解释率", pad=15)
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'pca_variance.png'))
    plt.show()

    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                          cmap='viridis', s=50, alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', s=200, marker='X', label='Cluster Centers')
    plt.title(f"K均值聚类结果 (聚类数={n_clusters})", pad=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'kmeans_clustering.png'))
    plt.show()

    return X_pca, clusters

# 执行PCA和聚类
X_pca, clusters = perform_pca_and_clustering(df)

# 4. 分类任务 (SVM)
def perform_classification(data, target_col='temperature_2m'):
    """执行分类任务"""
    # 创建分类标签（基于温度四分位数）
    quantiles = data[target_col].quantile([0.25, 0.5, 0.75]).values
    labels = ['cold', 'mild', 'warm', 'hot']
    data['temp_category'] = pd.cut(data[target_col],
                                   bins=[-np.inf] + list(quantiles) + [np.inf],
                                   labels=labels)

    # 准备数据
    X = data.drop(columns=[target_col, 'temp_category'])
    y = data['temp_category']

    # 创建预处理管道
    preprocessor = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 使用网格搜索优化SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    svm = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    svm.fit(preprocessor.fit_transform(X_train), y_train)

    # 评估模型
    y_pred = svm.predict(preprocessor.transform(X_test))

    print("\n SVM最佳参数:", svm.best_params_)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("支持向量机 混淆矩阵", pad=15)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix.png'))
    plt.show()

# 执行分类
perform_classification(df.copy())

# 5. 回归预测
def perform_regression(data, target_col='temperature_2m'):
    """执行回归预测"""
    # 准备数据
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 创建预处理管道
    preprocessor = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        PCA(n_components=0.95)
    )

    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 初始化模型
    models = {
        '线性回归': LinearRegression(),
        'SVR回归': SVR(kernel='rbf')
    }

    # 训练和评估模型
    results = {}
    for name, model in models.items():
        # 创建完整管道
        pipeline = make_pipeline(
            preprocessor,
            model
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # 存储结果
        results[name] = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'predictions': y_pred
        }

        print(f"\n {name} 性能:")
        print(f"MSE: {results[name]['mse']:.3f}")
        print(f"R²: {results[name]['r2']:.3f}")

    # 可视化预测结果
    plt.figure(figsize=(14, 6))
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(1, 2, i)
        plt.scatter(y_test, result['predictions'], alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(name, pad=10)
        plt.xlabel("Actual Temperature")
        plt.ylabel("Predicted Temperature")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'regression_results.png'))
    plt.show()

# 执行回归
perform_regression(df.copy())

print("\n 所有分析已完成！结果已保存到data文件夹")