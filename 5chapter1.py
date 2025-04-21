import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 读取Excel文件
file_path = "data/student_grade.xlsx"
df = pd.read_excel(file_path)

# 定义总成绩的区间和标签
bins = [0, 150, 200, 250, 300]  # 根据数据划分区间
labels = ['不及格', '及格', '良好', '优秀']

# 添加成绩区间分类列2

df['总成绩区间'] = pd.cut(df['总成绩'], bins=bins, labels=labels, right=False)

# 统计各个区间的人数
score_distribution = df['总成绩区间'].value_counts()

#饼图
plt.figure(figsize=(7, 7))
colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold']
explode = [0.025, 0.025, 0.025,0.025]

score_distribution.plot.pie(
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
)

plt.title("学生考试总成绩分布")
plt.ylabel("")  # 去掉默认的Y轴标签
plt.axis('equal')  # 保持饼图比例，避免变形
plt.show()

#绘制箱线
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['数学成绩', '阅读成绩', '写作成绩']],
            width=0.6,
            linewidth=2.5,
            showmeans=True,
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'red'})
plt.title("学生考试单科成绩分布", fontsize=14)
plt.xlabel("科目", fontsize=12)
plt.ylabel("成绩", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 计算分组均值
eff_mean = df.groupby('自我效能感')['总成绩'].mean().reindex(['低', '中', '高'])
prep_mean = df.groupby('考试课程准备情况')['总成绩'].mean()

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 自我效能感柱形图
colors = ['#FF9999', '#66B2FF', '#99FF99']
eff_mean.plot(kind='bar', ax=ax1, color=colors, edgecolor='black', width=0.7)
ax1.set_title('自我效能感对总成绩的影响', pad=20)
ax1.set_xlabel('自我效能感等级', labelpad=10)
ax1.set_ylabel('平均总成绩', labelpad=10)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(eff_mean):
    ax1.text(i, v+3, f"{v:.1f}", ha='center', fontweight='bold')

# 准备情况柱形图
prep_mean.plot(kind='bar', ax=ax2, color=['#FFCC99', '#CC99FF'], edgecolor='black', width=0.6)
ax2.set_title('课程准备情况对总成绩的影响', pad=20)
ax2.set_xlabel('准备情况', labelpad=10)
ax2.set_ylabel('平均总成绩', labelpad=10)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(prep_mean):
    ax2.text(i, v+3, f"{v:.1f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

#分析结果输出
print("\n=== 特征影响分析 ===")
print(f"不同自我效能感的平均总成绩：\n{eff_mean}")
print(f"\n不同准备情况的平均总成绩差异：")
print(f"完成准备的学生比未完成的平均高 {prep_mean['完成']-prep_mean['未完成']:.1f} 分")
