import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

we_18 = pd.read_csv("./data/weather/2018.csv", encoding="SHIFT-JIS", skiprows=4).drop(index=0).iloc[:,[0,1,4,7,11]]
we_19 = pd.read_csv("./data/weather/2019.csv", encoding="SHIFT-JIS", skiprows=4).drop(index=0).iloc[:,[0,1,4,7,11]]
we_20 = pd.read_csv("./data/weather/2020.csv", encoding="SHIFT-JIS", skiprows=4).drop(index=0).iloc[:,[0,1,4,7,11]]
we_21 = pd.read_csv("./data/weather/2021.csv", encoding="SHIFT-JIS", skiprows=4).drop(index=0).iloc[:,[0,1,4,7,11]]

#print(df_2018.head(5))

weather_data = pd.concat([we_18, we_19, we_20, we_21], ignore_index=True).reindex(columns=["年月日時", "気温(℃)", "降水量(mm)", "天気", "相対湿度(％)"])

#print(weather_data.head)

start_num = 0
weather_list = []
for _ in range(int(weather_data.shape[0]/24)):
    tmp_df = weather_data.iloc[start_num: start_num+24, 3]
    tmp_df.reset_index(drop=True, inplace=True)
    for _ in range(4):
        weather_list.append(tmp_df[2])
    for _ in range(3):
        weather_list.append(tmp_df[5])
    for _ in range(3):
        weather_list.append(tmp_df[8])
    for _ in range(3):
        weather_list.append(tmp_df[11])
    for _ in range(3):
        weather_list.append(tmp_df[14])
    for _ in range(3):
        weather_list.append(tmp_df[17])
    for _ in range(5):
        weather_list.append(tmp_df[20])
    start_num += 24

weather_data["天気"] = weather_list

weather_data["年月日時"] = weather_data["年月日時"].map(lambda x : dt.datetime.strptime(x, "%Y/%m/%d %H:%M:%S"))
weather_data.rename(columns={"年月日時":"datetime"}, inplace=True)

#========================================================================

data_18 = pd.read_csv("./data/power_demand/juyo-2018.csv", encoding="SHIFT-JIS", skiprows=2)
data_19 = pd.read_csv("./data/power_demand/juyo-2019.csv", encoding="SHIFT-JIS", skiprows=2)
data_20 = pd.read_csv("./data/power_demand/juyo-2020.csv", encoding="SHIFT-JIS", skiprows=2)
data_21 = pd.read_csv("./data/power_demand/juyo-2021.csv", encoding="SHIFT-JIS", skiprows=2)

data = pd.concat([data_18, data_19, data_20, data_21])
data.reset_index(drop=True, inplace=True)
data.rename(columns={"実績(万kW)":"kw"}, inplace=True)

tmp_list = []
for date,time in  zip(data["DATE"], data["TIME"]):
    strdate = date + " " + time
    tmp_list.append(dt.datetime.strptime(strdate, "%Y/%m/%d %H:%M"))
    
data["datetime"] = tmp_list

data["month"] = data["DATE"].map(lambda x : x.split("/")[1])
data["day"] = data["DATE"].map(lambda x : x.split("/")[-1])
data["hour"] = data["TIME"].map(lambda x : x.split(":")[0])
data["weekday"] = data["datetime"].map(lambda x : dt.datetime.weekday(x))

#========================================================================

df = pd.merge(data, weather_data, how="left", on="datetime")
df = df.iloc[1:, :].reset_index(drop=True)
df.rename(columns={"DATE":"date", "TIME":"time", "気温(℃)":"temperature", "降水量(mm)":"precipitation", "天気":"weather", "相対湿度(％)":"humidity"}, inplace=True)

#========
#前処理
#========

#天気一覧から辞書を作成
w_dic = {1:0.0, 2:0.0, 3:0.3, 4:0.5, 5:0.6, 6:1.0, 7:1.0, 8:0.6, 9:0.7, 10:1.0, 11:1.0,
         12:1.0, 13:1.0, 14:1.0, 15:1.0, 16:0.8, 17:1.0, 18:0.8, 19:1.0, 22:1.0, 23:1.0, 24:1.0, 28:0.6, 101:1.0}
# "weather"列を0から1の範囲に変換
df["weather"] = df["weather"].astype("int").map(lambda x: w_dic[x])

#欠損値を直前の値で補完
df = df.ffill()
df["weekday"] = df["weekday"].astype("object")
df["humidity"] = df["humidity"].map(lambda x : int(x)/100)

# df["humidity"].plot(figsize=(30,3))
# plt.show()
# df["temperature"].plot(figsize=(30,3))
# plt.show()
# df["kw"].plot(figsize=(30,3))
# plt.show()

#========
#train data 作成
#========

# "date", "time"を落とす
df = df.iloc[:, 2:]

# "kw"をtargetに、他をfeatureに
feature = df.iloc[:, 2:]
target = df.iloc[:, 0]

# 全体の2割をtestデータとして分割
train_size = round(feature.shape[0] * 0.8)
train_x = feature.iloc[:train_size, :]
test_x = feature.iloc[train_size:, :].reset_index(drop=True)
train_y = target.iloc[:train_size]
test_y = target.iloc[train_size:].reset_index(drop=True)

mms_x = MinMaxScaler()
mms_x.fit(train_x.iloc[:, 4:])
train_x.iloc[:, 4:] = mms_x.transform(train_x.iloc[:, 4:])
test_x.iloc[:, 4:] = mms_x.transform(test_x.iloc[:, 4:])

mms_y = MinMaxScaler()
mms_y.fit(train_y.values.reshape(-1,1))
train_y = mms_y.transform(train_y.values.reshape(-1,1))
test_y = mms_y.transform(test_y.values.reshape(-1,1))

train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)

# print(train_x.head())
# print(test_x.head())

# 前6時間分のデータを参照する
timesteps = 6

data_x = []
xarr = np.array
for i in range(timesteps, train_x.shape[0]):
    xset = []
    for j in range(train_x.shape[1]):
        d = train_x.iloc[i-timesteps:i, j]
        xset.append(d)
    xarr = np.array(xset).reshape(timesteps, train_x.shape[1])
    data_x.append(xarr)
x_train = np.array(data_x)

# 6件目以降のデータを目的変数に
y_train = train_y[timesteps:]

#testデータも同様に変換
data_x = []
xarr = np.array
for i in range(timesteps, test_x.shape[0]):
    xset = []
    for j in range(test_x.shape[1]):
        d = test_x.iloc[i-timesteps:i, j]
        xset.append(d)
    xarr = np.array(xset).reshape(timesteps, test_x.shape[1])
    data_x.append(xarr)
x_test = np.array(data_x)

# 6件目以降のデータを目的変数に
y_test = test_y[timesteps:]

#========
#モデル作成
#========

import time
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, SimpleRNN, GRU
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

neuron = 64
actfunc = "tanh"
dropout = 0.2
epochs = 300

#========
#RNN
#========

start = time.perf_counter() #計測開始

model_RNN_64 = Sequential()
model_RNN_64.add(Input(shape=(timesteps, x_train.shape[2])))
model_RNN_64.add(SimpleRNN(neuron, activation=actfunc, return_sequences=False))
model_RNN_64.add(Dropout(dropout))
model_RNN_64.add(Dense(1, activation="linear")) 

model_RNN_64.compile(loss="mean_squared_error", optimizer="adam")
early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3)

history_RNN_64 = model_RNN_64.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

end = time.perf_counter() #計測終了
print(end-start)

plt.figure(figsize=(15,8))
plt.plot(history_RNN_64.history['loss'], label='Train Loss')
plt.plot(history_RNN_64.history['val_loss'], label='valid Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()

#========
#GRU
#========

start = time.perf_counter() #計測開始

model_GRU_64 = Sequential()
model_GRU_64.add(Input(shape=(timesteps, x_train.shape[2])))
model_GRU_64.add(GRU(neuron, activation=actfunc, return_sequences=False))
model_GRU_64.add(Dropout(dropout))
model_GRU_64.add(Dense(1, activation="linear")) 

model_GRU_64.compile(loss="mean_squared_error", optimizer="adam")
early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3)

history_GRU_64 = model_GRU_64.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

end = time.perf_counter() #計測終了
print(end-start)

plt.figure(figsize=(15,8))
plt.plot(history_GRU_64.history['loss'], label='Train Loss')
plt.plot(history_GRU_64.history['val_loss'], label='valid Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()

#========
#LSTM
#========

start = time.perf_counter() #計測開始

model_LSTM_64 = Sequential()
model_LSTM_64.add(Input(shape=(timesteps, x_train.shape[2])))
model_LSTM_64.add(LSTM(neuron, activation=actfunc, return_sequences=False))
model_LSTM_64.add(Dropout(dropout))
model_LSTM_64.add(Dense(1, activation="linear")) 

model_LSTM_64.compile(loss="mean_squared_error", optimizer="adam")
early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3)

history_LSTM_64 = model_LSTM_64.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

end = time.perf_counter() #計測終了
print(end-start)

plt.figure(figsize=(15,8))
plt.plot(history_LSTM_64.history['loss'], label='Train Loss')
plt.plot(history_LSTM_64.history['val_loss'], label='valid Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()

#========
#モデル評価
#========

#========
#RNN
#========

y_test_pred_RNN_64 = model_RNN_64.predict(x_test)
y_test_pred_RNN_64 = mms_y.inverse_transform(y_test_pred_RNN_64)
y_t = mms_y.inverse_transform(y_test)

RNN_RMSE_64 = np.sqrt(mean_squared_error(y_t, y_test_pred_RNN_64))
RNN_MAE_64 = mean_absolute_error(y_t, y_test_pred_RNN_64)
RNN_MAPE_64 = mean_absolute_percentage_error(y_t, y_test_pred_RNN_64)

# 指標出力
print('RMSE:')
print(RNN_RMSE_64)
print('MAE:')
print(RNN_MAE_64) 
print('MAPE:')
print(RNN_MAPE_64) 

#テストデータ(青)と予測(黄色)を描画
plt.figure(figsize=(30,8))
plt.plot(y_t, c="b")
plt.plot(y_test_pred_RNN_64, c="y")
plt.show()

#========
#GRU
#========

y_test_pred_GRU_64 = model_GRU_64.predict(x_test)
y_test_pred_GRU_64 = mms_y.inverse_transform(y_test_pred_GRU_64)
y_t = mms_y.inverse_transform(y_test)

GRU_RMSE_64 = np.sqrt(mean_squared_error(y_t, y_test_pred_GRU_64))
GRU_MAE_64 = mean_absolute_error(y_t, y_test_pred_GRU_64)
GRU_MAPE_64 = mean_absolute_percentage_error(y_t, y_test_pred_GRU_64)

# 指標出力
print('RMSE:')
print(GRU_RMSE_64)
print('MAE:')
print(GRU_MAE_64) 
print('MAPE:')
print(GRU_MAPE_64) 

#テストデータ(青)と予測(黄色)を描画
plt.figure(figsize=(30,8))
plt.plot(y_t, c="b")
plt.plot(y_test_pred_GRU_64, c="y")
plt.show()

#========
#LSTM
#========

y_test_pred_LSTM_64 = model_LSTM_64.predict(x_test)
y_test_pred_LSTM_64 = mms_y.inverse_transform(y_test_pred_LSTM_64)
y_t = mms_y.inverse_transform(y_test)

LSTM_RMSE_64 = np.sqrt(mean_squared_error(y_t, y_test_pred_LSTM_64))
LSTM_MAE_64 = mean_absolute_error(y_t, y_test_pred_LSTM_64)
LSTM_MAPE_64 = mean_absolute_percentage_error(y_t, y_test_pred_LSTM_64)

# 指標出力
print('RMSE:')
print(LSTM_RMSE_64)
print('MAE:')
print(LSTM_MAE_64) 
print('MAPE:')
print(LSTM_MAPE_64) 

#テストデータ(青)と予測(黄色)を描画
plt.figure(figsize=(30,8))
plt.plot(y_t, c="b")
plt.plot(y_test_pred_LSTM_64, c="y")
plt.show()

# 後ろから2週間分(24*7*2)のデータだけ描画
plt.figure(figsize=(30,8))
plt.plot(y_t[-336:], c="b")
plt.plot(y_test_pred_RNN_64[-336:], c="g")
plt.plot(y_test_pred_GRU_64[-336:], c="y")
plt.plot(y_test_pred_LSTM_64[-336:], c="r")
plt.show()