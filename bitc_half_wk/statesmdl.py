import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据.csv")

# 将日期列转换为数值型
df['日期'] = pd.to_datetime(df['日期'])
df['日期_numeric'] = pd.to_numeric(df['日期'])

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
    delta4 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2025-06-17')).days
    df.at[c, '减半事件'] = min(delta1, delta2, delta3, delta4)


decay_parameter = 0.001
weights = np.exp(-decay_parameter * np.array(df['减半事件']))
normalized_weights = weights / np.sum(weights)
# 创建一个DataFrame，以便更好地展示数据
df['weights'] = normalized_weights * 100

# for i in df['weights']:
#     print(i)

# 去除百分比符号 '%' 并将 '涨跌幅' 列转换为浮点数
df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype('float') / 100.0

df['开盘'] = df['开盘'].str.replace(',', '').astype('float')
df['收盘'] = df['收盘'].str.replace(',', '').astype('float')
df['高'] = df['高'].str.replace(',', '').astype('float')
df['低'] = df['低'].str.replace(',', '').astype('float')

# 添加比特币减半事件的连续变量
df['减半事件'] = (abs(pd.to_datetime(df['日期']) - pd.to_datetime('2012-11-28')) + abs(pd.to_datetime(df['日期']) -
                  pd.to_datetime('2016-07-09')) + abs(pd.to_datetime(df['日期']) - pd.to_datetime('2020-05-11'))).dt.days
df['减半事件'] /= df['减半事件'].max()  # 缩放到 [0, 1]

columns_to_save = ['日期', '开盘', '收盘', '减半事件', 'weights']
df[columns_to_save].to_csv('processed_bitcoin_data.csv', index=False)

# # 选择自变量和因变量
# X = df[['日期_numeric', '收盘', '开盘', '高', '低', '涨跌幅', '减半事件']]
# Y = df['涨跌幅']
#
# # 添加截距项
# X = sm.add_constant(X)
#
# # 拟合回归模型
# model = sm.OLS(Y, X).fit()
#
# # 打印回归结果
# print(model.summary())
#
# # 绘制实际涨跌幅和模型拟合结果的对比图
# plt.figure(figsize=(12, 6))
# plt.plot(df['日期'], df['涨跌幅'], label='Actual Returns')
# plt.plot(df['日期'], model.fittedvalues, label='Fitted Returns')
# plt.title('Actual vs Fitted Returns')
# plt.xlabel('日期')
# plt.ylabel('涨跌幅')
# plt.legend()
# plt.show()

# 选择自变量和因变量
# X = df[['减半事件', '开盘']]
# Y = df['收盘']

# 拆分数据集为训练集和测试集
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 按照日期对数据集进行拆分
split_date = pd.to_datetime('2022-02-01')  # 选择拆分日期
train_data = df[df['日期'] < split_date]
test_data1 = df[df['日期'] >= split_date]
split_date1 = pd.to_datetime('2024-3-02')  # 选择拆分日期
test_data = test_data1[test_data1['日期'] < split_date1]

# 定义自变量和因变量
X_train, X_test = train_data[['开盘']], test_data[['开盘']]
X, Y = train_data[['日期']], test_data[['日期']]
Y_train, Y_test = train_data['收盘'], test_data['收盘']

model = RandomForestRegressor(n_estimators=80, random_state=42)
# model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 120, random_state = 42)
model.fit(X_train, Y_train)

# 预测结果
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(Y_test, test_predictions)
print('MSE:', mse)

# 计算均方根误差
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# 计算平均绝对误差
mae = mean_absolute_error(Y_test, test_predictions)
print('MAE:', mae)

# for i in test_predictions:
#     print(i)

# print(test_predictions)

# 绘制实际涨跌幅和预测结果的对比图
# plt.figure(figsize=(12, 6))
# plt.plot(X, Y_train, label='Actual Returns')
# plt.plot(X, train_predictions, label='Predicted Returns')
# plt.title('Actual vs Predicted Returns')
# plt.xlabel('日期')
# plt.ylabel('收盘')
# plt.legend()
# plt.show()

# 绘制实际涨跌幅和预测结果的对比图
plt.figure(figsize=(12, 6))
plt.plot(Y, Y_test, label='True closing price')
plt.plot(Y, test_predictions, label='Predicted closing price')
# plt.title('Actual vs Predicted Returns')
plt.xlabel('Date')
plt.ylabel('Closing price')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(Y_test, test_predictions, color='blue')
# plt.title('Actual vs Predicted Returns')
# plt.xlabel('Actual Returns')
# plt.ylabel('Predicted Returns')
# plt.show()


# # 绘制实际涨跌幅和预测结果的对比图（训练集）
# plt.figure(figsize=(12, 6))
# plt.plot(X_train.index, Y_train, label='Actual Returns (Train)')
# plt.plot(X_train.index, train_predictions, label='Predicted Returns (Train)')
# plt.title('Actual vs Predicted Returns (Training Set)')
# plt.xlabel('日期')
# plt.ylabel('涨跌幅')
# plt.legend()
# plt.show()
#
# # 绘制实际涨跌幅和预测结果的对比图（测试集）
# plt.figure(figsize=(12, 6))
# plt.plot(X_test.index, Y_test, label='Actual Returns (Test)')
# plt.plot(X_test.index, test_predictions, label='Predicted Returns (Test)')
# plt.title('Actual vs Predicted Returns (Test Set)')
# plt.xlabel('日期')
# plt.ylabel('涨跌幅')
# plt.legend()
# plt.show()
