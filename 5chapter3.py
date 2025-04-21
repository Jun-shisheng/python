import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Funnel, WordCloud

# 读取数据
file_path = "data/商品销售数据.csv"
df = pd.read_csv(file_path, header=None, names=[
    "订单号", "设备ID", "应付金额", "实际金额", "商品", "支付时间",
    "地点", "状态", "提现", "大类", "二级类"
])

# 数据清洗
df["实际金额"] = pd.to_numeric(df["实际金额"], errors="coerce")  # 转换为数值类型
df = df.dropna(subset=["实际金额"])  # 删除缺失值

# 统计销售额（按二级类别）
category_sales = df.groupby("二级类")["实际金额"].sum().sort_values(ascending=False)

# 统计商品销量（按商品名称）
product_counts = df["商品"].value_counts()

# 选取销售额前5的商品类别
top5_sales = category_sales.head(5)

# 绘制漏斗图（前5销售额类别）
funnel = (
    Funnel()
    .add(
        "销售额",
        [list(z) for z in zip(top5_sales.index, top5_sales.values)],
        sort_="descending",
        label_opts=opts.LabelOpts(position="inside"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="销售额前5的商品类别漏斗图"),
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b}: {c}元"),
    )
)

# 将商品销量转换为适合 WordCloud 的格式
wordcloud_data =[(k,v) for k,v in product_counts.items()]

# 绘制词云图（商品销量）
wordcloud = (
    WordCloud()
    .add(
        "商品销量",
        wordcloud_data,
        word_size_range=[12, 60],
        shape="circle",
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="商品销量词云图"),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
)

# 生成 HTML 文件路径
funnel_path = "data/sales_funnel.html"
wordcloud_path = "data/product_wordcloud.html"

# 渲染并保存为 HTML 文件
funnel.render(funnel_path)
wordcloud.render(wordcloud_path)

print("图表已生成.")
