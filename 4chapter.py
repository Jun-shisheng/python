import pandas as pd

# 读取文件
stroke_df = pd.read_excel('data/healthcare-dataset-stroke.xlsx')
age_blood_df = pd.read_excel('data/healthcare-dataset-age_abs.xlsx')

# 合并
merged_df = pd.merge(stroke_df, age_blood_df, on='编号', how='outer')

# 重置索引
merged_df.reset_index(drop=True, inplace=True)

# 删除年龄异常值
delete_indices = []
for i in range(len(merged_df)):
    age = merged_df.loc[i, '年龄/岁']
    if pd.isnull(age) or age <= 0 or age > 120:
        delete_indices.append(i)

merged_df.drop(index=delete_indices, inplace=True)
merged_df.reset_index(drop=True, inplace=True)

# 自定义分箱边界和标签
bin_edges = [0, 20, 40, 60, 80, 120]
labels = [f'{bin_edges[i]}-{bin_edges[i+1]-1}岁' for i in range(len(bin_edges)-1)]

# 等宽离散化，使用自定义整数区间
merged_df['年龄分段'] = pd.cut(
    merged_df['年龄/岁'],
    bins=bin_edges,
    labels=labels,
    right=False
)

# 打印区间信息（格式对齐）
print("\n年龄区间范围:")
for label in labels:
    print(f"{'区间'.ljust(10)}: {label}")

# 查看处理后的数据（对齐输出）
print("\n处理异常值和离散化后的数据（merged_df）:")
print(merged_df[['编号', '年龄/岁', '年龄分段']].to_string(index=False))

# 统计区间人数
count_series = merged_df['年龄分段'].value_counts().sort_index()

# 打印区间人数（格式对齐）
print("\n每个年龄区间的人数统计:")
print(f"{'年龄分段'.ljust(15)} | {'人数'}")
print("-" * 25)
for index, value in count_series.items():
    print(f"{index.ljust(15)} | {value}")

# 保存结果到 data 文件夹
output_file = 'data/merged_result.xlsx'
merged_df.to_excel(output_file, index=False)

print(f"\n数据已成功保存到文件: {output_file}")
