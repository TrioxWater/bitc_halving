import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df1 = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据 (1).csv")

# 将 'Date' 列转换为日期格式
df1['日期'] = pd.to_datetime(df1['日期'])
df1 = df1.sort_values('日期')
# 去除百分比符号 '%' 并将 '涨跌幅' 列转换为浮点数
df1['涨跌幅'] = df1['涨跌幅'].str.rstrip('%').astype('float') / 100.0

df1['开盘'] = df1['开盘'].str.replace(',', '').astype('float')
df1['收盘'] = df1['收盘'].str.replace(',', '').astype('float')
df1['高'] = df1['高'].str.replace(',', '').astype('float')
df1['低'] = df1['低'].str.replace(',', '').astype('float')

# 选择特征和目标
features = ["收盘", "开盘", "高", "低"]
target = "涨跌幅"  # 或者使用 "交易量"

# 提取特征和目标列
X = df1[features]
y = df1[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立 MLP 模型
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# 训练模型
model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print("均方误差 (MSE):", mse)

# 可视化预测结果
plt.plot(y_test.values, label="实际值")
plt.plot(y_pred, label="预测值")
plt.legend()
plt.show()
