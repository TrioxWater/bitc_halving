import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# 假设你已经有了比特币价格数据（price_btc）和控制组数据（price_control），以及每次减半的日期（halving_date）。
# 这里只是一个简单的示例，实际数据可能会更加复杂
halving_dates = ['2012-11-28', '2016-07-09', '2020-05-12']

# 创建一个示例数据框
data = {
    'date': pd.date_range(start='2012-01-01', end='2022-01-01', freq='D'),  # 假设数据从2012年1月1日到2022年1月1日
    'price_btc': np.random.rand(3650),  # 3650天的比特币价格数据，这里只是随机生成的示例数据
    'price_control': np.random.rand(3650),  # 3650天的控制组价格数据，这里只是随机生成的示例数据
}

df = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据 (1).csv")

# 将日期列转换为数值型
df['日期'] = pd.to_datetime(df['日期'])
# 去除百分比符号 '%' 并将 '涨跌幅' 列转换为浮点数
df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype('float') / 100.0

df1 = df[(df['日期'] > pd.to_datetime('2014-07-09')) & (df['日期'] <= pd.to_datetime('2017-01-09'))]
df2 = df[(df['日期'] > pd.to_datetime('2017-01-09')) & (df['日期'] <= pd.to_datetime('2019-07-09'))]

btc_price = df1['涨跌幅']
control_price = df2['涨跌幅']

# 计算比特币价格和控制组价格的均值
btc_mean = btc_price.mean()
control_mean = control_price.mean()

# 使用t检验检验均值差异的显著性
t_stat, p_value = ttest_ind(btc_price, control_price)

# 输出结果
print(f"比特币价格均值: {btc_mean}")
print(f"控制组价格均值: {control_mean}")
print(f"t统计量: {t_stat}, p值: {p_value}")
print("---------------")

# # 创建一个事件窗口
# event_window = 120  # 假设事件窗口为减半日期前后的30天
#
# # 分析每次减半事件的影响
# for halving_date in halving_dates:
#     start_date = halving_date - pd.Timedelta(days=event_window)
#     end_date = halving_date + pd.Timedelta(days=event_window)
#
#     # 选择事件窗口内的数据
#     event_data = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
#     btc_price = event_data['btc_price']
#     control_price = event_data['price_control']
#
#     # 计算比特币价格和控制组价格的均值
#     btc_mean = btc_price.mean()
#     control_mean = control_price.mean()
#
#     # 使用t检验检验均值差异的显著性
#     t_stat, p_value = ttest_ind(btc_price, control_price)
#
#     # 输出结果
#     print(f"减半日期: {halving_date}")
#     print(f"比特币价格均值: {btc_mean}")
#     print(f"控制组价格均值: {control_mean}")
#     print(f"t统计量: {t_stat}, p值: {p_value}")
#     print("---------------")
