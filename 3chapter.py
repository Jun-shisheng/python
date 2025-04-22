import pandas as pd

# 禁用科学计数法显示，保留两位小数
# pd.options.display.float_format = '{:.2f}'.format

# 读取数据
data = pd.read_csv("data/house.csv")
print("房屋数据读取成功！")

# 预览前5行数据
print("\n房屋数据预览（前5行）：")
print(data.head())

# 显示数据的维度
print("\n数据维度（ndim）：")
print(f"该数据是 {data.ndim} 维结构。")

# 显示数据的形状
print("\n数据形状（shape）：")
print(f"该数据有 {data.shape[0]} 行，{data.shape[1]} 列。")

# 显示所有列名
print("\n列名（columns）：")
print(data.columns.tolist())

print("\n单列房屋数据索引（iloc 方法，第4列）：")
if data.shape[1] >= 4:
    data1 = data.iloc[:, [4]]
    print(data1)
else:
    print("数据中列数不足，无法提取第4列！")

print("\n根据列名索引（loc 方法，'房屋类型' 列）：")
if '房屋类型' in data.columns:
    data2 = data.loc[:, ['房屋类型']]
    print(data2)
else:
    print("列名 '房屋类型' 不存在，请确认列名是否正确！")

if '房屋出售时间' in data.columns:
    print("\n将 '房屋出售时间' 列转换为日期格式（to_datetime）：")
    data['房屋出售时间'] = pd.to_datetime(data['房屋出售时间'], errors='coerce')
    print(data['房屋出售时间'].head())
else:
    print("\n没有找到 '房屋出售时间' 列，无法进行日期转换。")

if '房屋价格/元' in data.columns:
    print("\n房价数据统计：")

# 平均值
    mean_price = data['房屋价格/元'].mean()
    print(f"平均价格（mean）：{mean_price:.2f} 元")

# 最大值
    max_price = data['房屋价格/元'].max()
    print(f"最高价格（max）：{max_price:.2f} 元")

# 最小值
    min_price = data['房屋价格/元'].min()
    print(f"最低价格（min）：{min_price:.2f} 元")

# 众数
    mode_price = data['房屋价格/元'].mode()
    print(f"众数（mode）：{[round(m, 2) for m in mode_price.tolist()]} 元")

# 分位数
    print("\n分位数（quantile）：")
    q1 = data['房屋价格/元'].quantile(0.25)
    q2 = data['房屋价格/元'].quantile(0.5)
    q3 = data['房屋价格/元'].quantile(0.75)
    print(f"25% 分位数：{q1:.2f} 元")
    print(f"50% 分位数（中位数）：{q2:.2f} 元")
    print(f"75% 分位数：{q3:.2f} 元")

    desc_stats = data['房屋价格/元'].describe()
    count = desc_stats['count']
    mean = desc_stats['mean']
    print("\n下列是非空数据量和均值：")
    print(f"非空数据数量（count）：{int(count)}")
    print(f"均值（mean）：{mean:.2f} 元")

else:
    print("\n没有找到 '房屋价格/元' 列，无法计算统计信息。")

# 生成 new_postcode 特征
if '地区邮编' in data.columns:
    print("\n正在基于 '地区邮编' 列创建 new_postcode 特征...")

    # 使用 apply() 方法生成新的邮编特征，假设只保留邮编的前4位
    data['new_postcode'] = data['地区邮编'].apply(lambda x: str(x)[:4])

    print("new_postcode 特征生成完成，前5行如下：")
    print(data[['地区邮编', 'new_postcode']].head())
else:
    print("\n没有找到 '地区邮编' 列，无法生成 new_postcode 特征。")

# 使用 agg() 和 count() 统计每个地区的房屋售出总数（假设有 '地区' 列）
if '地区邮编' in data.columns:
    print("\n正在按地区统计房屋售出总数...")

    # 按地区分组，计算每个地区的房屋数量
    region_house_count = data.groupby('地区邮编').agg(售出总数=('地区邮编', 'count'))

    print("每个地区房屋售出总数统计结果：")
    print(region_house_count)
else:
    print("\n没有找到 '地区' 列，无法统计每个地区的房屋售出总数。")

# 使用 groupby 方法按 '房屋类型' 和 'new_postcode' 分组
if '房屋类型' in data.columns and 'new_postcode' in data.columns:
    print("\n正在按 '房屋类型' 和 'new_postcode' 进行分组...")

    # 进行分组，并赋值给新的数据框 housesale1
    housesale1 = data.groupby(['房屋类型', 'new_postcode'])

    print("分组完成，显示分组对象信息：")
    print(housesale1)

# 使用 transform 和 mean 计算分组后的均值
    if '房屋价格/元' in data.columns:
        # 创建一个新的列，保存分组均值
        data['分组房价均值'] = housesale1['房屋价格/元'].transform('mean')

        print("分组均值计算完成，前5行展示如下：")
        print(data[['房屋类型', 'new_postcode', '房屋价格/元', '分组房价均值']].head())
    else:
        print("\n没有找到 '房屋价格/元' 列，无法计算分组均值。")
else:
    print("\n没有找到 '房屋类型' 或 'new_postcode' 列，无法进行分组。")

# 使用 pivot_table 创建透视表
if '房屋类型' in data.columns and 'new_postcode' in data.columns and '房屋价格/元' in data.columns:
    print("\n正在使用 pivot_table 创建透视表...")

    # 创建透视表：按 房屋类型 和 new_postcode 分组，统计平均价格
    pivot = pd.pivot_table(
        data,
        values='房屋价格/元',
        index='房屋类型',
        columns='new_postcode',
        aggfunc='mean'
    )

    print("透视表创建完成，展示前几行数据：")
    print(pivot.head())
else:
    print("\n创建透视表失败，缺少 '房屋类型'、'new_postcode' 或 '房屋价格/元' 列。")

# 使用 crosstab 创建数据交叉表
if '房屋类型' in data.columns and 'new_postcode' in data.columns:
    print("\n正在使用 crosstab 创建交叉表...")

    # 交叉表：房屋类型 和 new_postcode 出现的次数频数统计
    cross = pd.crosstab(
        index=data['房屋类型'],
        columns=data['new_postcode'],
        margins=True,        # 增加总计行和列
        margins_name='总计'   # 总计列的名称
    )

    print("交叉表创建完成，展示前几行数据：")
    print(cross.head())
else:
    print("\n创建交叉表失败，缺少 '房屋类型' 或 'new_postcode' 列。")
