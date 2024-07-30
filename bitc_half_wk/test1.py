import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, MaxPooling1D, Conv1D
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
data = pd.read_csv("D:/PCCommunity/bitc_half_wk/processed_bitcoin_data.csv")
data['日期'] = pd.to_datetime(data['日期'])

# 格式化数据
X = data[["weights", "收盘"]]
y = data["收盘"]

# 将数据转换为三维数组
X = X.to_numpy().reshape(-1, 2, 1)
y = y.to_numpy().reshape(-1, 1)

df = data[['日期']]

# 划分训练集和测试集
train_X = X[int(0.1 * len(X)):]
train_y = y[int(0.1 * len(y)):]
train_date = df[int(0.1 * len(df)):]
test_X = X[:int(0.1 * len(X))]
test_y = y[:int(0.1 * len(y))]
test_date = df[:int(0.1 * len(df))]

# 创建模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128))
model.add(Dense(1))

# 编译模型
model.compile(loss="mse", optimizer=Adam())

# 训练模型
model.fit(train_X, train_y, epochs=50)

# 评估模型
score = model.evaluate(test_X, test_y, verbose=0)
print("Test loss:", score)

# 预测收盘价
predictions = model.predict(test_X)

# 打印预测结果
print(predictions[0])

# 计算均方误差
mse = mean_squared_error(test_y, predictions)
print('Mean Squared Error (MSE):', mse)

# 计算均方根误差
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# 计算平均绝对误差
mae = mean_absolute_error(test_y, predictions)
print('Mean Absolute Error (MAE):', mae)

# # 绘制实际涨跌幅和预测结果的对比图
# plt.figure(figsize=(12, 6))
# plt.plot(train_date, train_y, label='Actual Returns')
# plt.plot(train_date, predictions, label='Predicted Returns')
# plt.title('Actual vs Predicted Returns')
# plt.xlabel('日期')
# plt.ylabel('收盘')
# plt.legend()
# plt.show()

# 绘制实际涨跌幅和预测结果的对比图
plt.figure(figsize=(12, 6))
plt.plot(test_date, test_y, label='Actual Returns')
plt.plot(test_date, predictions, label='Predicted Returns')
plt.title('Actual vs Predicted Returns')
plt.xlabel('日期')
plt.ylabel('收盘')
plt.legend()
plt.show()
