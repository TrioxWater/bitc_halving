import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据 (1).csv")

# 解析日期列为日期时间对象
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values('日期')

num = len(df)

df['减半事件'] = 0
for c, i in df.iterrows():
    # if i['日期'] == pd.to_datetime('2012-11-28') or i['日期'] == pd.to_datetime('2016-07-09') or \
    #         i['日期'] == pd.to_datetime('2020-05-11'):
    #     df.at[c, '减半事件'] = 1
    # else:
    #     # i['减半事件'] = 1 / abs(pd.to_datetime(i['日期']) - pd.to_datetime('2012-11-28')).dt.days + \
    #     #     1 / abs(pd.to_datetime(i['日期']) - pd.to_datetime('2016-07-09')).dt.days + \
    #     #     1 / abs(pd.to_datetime(i['日期']) - pd.to_datetime('2020-05-11')).dt.days
    #
    #     delta1 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2012-11-28')).days
    #     delta2 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2016-07-09')).days
    #     delta3 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2020-05-11')).days
    #     df.at[c, '减半事件'] = 1 / delta1 + 1 / delta2 + 1 / delta3

    delta1 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2014-01-28')).days
    delta2 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2017-09-09')).days
    delta3 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2021-07-11')).days
    df.at[c, '减半事件'] = min(delta1, delta2, delta3)


decay_parameter = 0.001
weights = np.exp(-decay_parameter * np.array(df['减半事件']))
normalized_weights = weights / np.sum(weights)
# 创建一个DataFrame，以便更好地展示数据
df['normalized_weights'] = normalized_weights

# # 添加比特币减半事件的连续变量
# df['减半事件'] = (abs(pd.to_datetime(df['日期']) - pd.to_datetime('2012-11-28')) + abs(pd.to_datetime(df['日期']) -
#                   pd.to_datetime('2016-07-09')) + abs(pd.to_datetime(df['日期']) -
#                   pd.to_datetime('2020-05-11'))).dt.days
# df['减半事件'] /= df['减半事件'].max()  # 缩放到 [0, 1]

for i in df['减半事件']:
    print(i)

# 去除百分比符号 '%' 并将 '涨跌幅' 列转换为浮点数
df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype('float') / 100.0

df['开盘'] = df['开盘'].str.replace(',', '').astype('float')
df['收盘'] = df['收盘'].str.replace(',', '').astype('float')

# 选择自变量和因变量
X = df[['减半事件', '开盘', '日期']]
Y = df['收盘']

# 拆分数据集为训练集和测试集
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 按照日期对数据集进行拆分
split_date = pd.to_datetime('2022-01-01')  # 选择拆分日期
train_data = df[df['日期'] < split_date]
test_data = df[df['日期'] >= split_date]

# 定义自变量和因变量
X_train, X_test = train_data[['normalized_weights', '开盘', '日期']], test_data[['normalized_weights', '开盘', '日期']]
Y_train, Y_test = train_data['收盘'], test_data['收盘']

# 使用决策树回归模型
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train[['normalized_weights', '开盘']], Y_train)

# 预测结果
predictions_train = model.predict(X_train[['normalized_weights', '开盘']])
predictions_test = model.predict(X_test[['normalized_weights', '开盘']])

# print(Y_train)
# for i in predictions_test:
#     print(i)

# # 绘制训练集中实际涨跌幅和预测结果的对比图
# plt.figure(figsize=(12, 6))
# plt.plot(X_train['日期'], Y_train, marker='o', linestyle='-', color='blue', label='Actual Returns (Training)')
# plt.plot(X_train['日期'], predictions_train, marker='o', linestyle='-', color='red',
#          label='Predicted Returns (Training)')
# plt.title('Actual vs Predicted Returns (Training)')
# plt.xlabel('日期')
# plt.ylabel('涨跌幅')
# plt.legend()
# plt.grid(True)
# plt.show()

# 绘制测试集中实际涨跌幅和预测结果的对比图
plt.figure(figsize=(12, 6))
plt.plot(X_test['日期'], Y_test,  linestyle='-', color='blue', label='Actual Returns (Testing)')
plt.plot(X_test['日期'], predictions_test,  linestyle='-', color='green', label='Predicted Returns (Testing)')
plt.title('Actual vs Predicted Returns (Testing)')
plt.xlabel('日期')
plt.ylabel('收盘')
plt.legend()
plt.grid(True)
plt.show()
