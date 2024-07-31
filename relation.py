import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os

if os.path.exists('DigitCur-USD'):
    os.chdir('DigitCur-USD')

# 读取csv文件
df = pd.read_csv('BTC-USD.csv')
# 币价取开盘价
price = df['Open']
# 日期，%Y/%m/%d
date = df['Date']
date = [time.strptime(i, '%Y/%m/%d') for i in date]

# dPrice = [price[i] - price[i - 1] for i in range(1, len(price))]
dPrice  = price

bitcoin_halving_date = ['2012-11-28', '2016-7-9', '2020-5-11']
bitcoin_halving_date = [time.strptime(i, '%Y-%m-%d') for i in bitcoin_halving_date]

bitcoin_miner_reward = [50 if i < bitcoin_halving_date[0] else 25 if i < bitcoin_halving_date[1] else 12.5 for i in date]

# 相关性核验
# 1. 价格与挖矿奖励

best = 0
bestTime = 0
# Pearson相关系数
for delayTime in range(1, 90):
    # 认为挖矿奖励的反馈具有滞后性，所以需要延迟一段时间
    p = np.corrcoef(bitcoin_miner_reward[:-delayTime], dPrice[delayTime:])[0][1]
    # print('挖矿奖励与币价的相关性滞后%d天，相关系数为%f' % (delayTime, p))
    if abs(p) > best:
        best = abs(p)
        bestTime = delayTime

print('挖矿奖励与币价的相关性最好的滞后时间为%d天，相关系数为%f' % (bestTime, best))









