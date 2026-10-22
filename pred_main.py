import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

ds_msm = xr.open_dataset("data/msm-s_auto/all.nc", cache=False)

df_pv = pd.read_csv(
    "data/power_generation/pvoutput/112207/processed/112207_pro_20260326-20260729.csv",
    encoding="utf-8-sig",
    index_col="datetime",
    parse_dates=["datetime"],
    )
df_pv.index = df_pv.index.tz_localize("Asia/Tokyo")

ds_msm_sliced = ds_msm.assign_coords(time = ds_msm.time + pd.Timedelta(hours=9))

ds_msm_sliced = ds_msm_sliced.sel(
	time = slice(np.datetime64("2026-03-26 00:00:00"), np.datetime64("2026-07-30 00:00:00")),
	latitude = 35.7999995,
	longitude = 139.0625,
    )

df_msm = ds_msm_sliced.to_dataframe()
df_msm.index = df_msm.index.tz_localize("Asia/Tokyo")

df_pv_1h = df_pv.resample("1h").sum()

df_all = df_pv_1h.join(df_msm)

df_all.to_csv("data/all.csv")

#========================================================================================

# df_all["hour"] = df_all.index.hour.astype(np.float32)
# df_all["hour_sin"] = np.sin(2*np.pi*df_all["hour"]/24)
# df_all["hour_cos"] = np.cos(2*np.pi*df_all["hour"]/24)

from astral import LocationInfo
from astral.sun import sun

location = LocationInfo( #奥多摩を参照
    latitude=35.7999995,
    longitude=139.0625,
    timezone="Asia/Tokyo"
)

sun_times = {
    date: sun(
        location.observer,
        date=date,
        tzinfo=location.timezone
    )
    for date in df_all.index.date
}

df_all["daylight"] = [
    int(
        sun_times[t.date()]["sunrise"] <= t
        < sun_times[t.date()]["sunset"]
    )
    for t in df_all.index
]

# df_all = df_all.drop(columns="hour")

# print(df_all)

feature = df_all.iloc[:, 2:]
target = df_all.iloc[:, 0]

# 全体の2割をtestデータとして分割
train_size = round(feature.shape[0] * 0.8)
train_x = feature.iloc[:train_size, :]
test_x = feature.iloc[train_size:, :].reset_index(drop=True)
train_y = target.iloc[:train_size]
test_y = target.iloc[train_size:].reset_index(drop=True)

mms_x = MinMaxScaler()
mms_x.fit(train_x.iloc[:, 4:])

train_x.iloc[:, 4:] = mms_x.transform(train_x.iloc[:, 4:]).astype(np.float32)
test_x.iloc[:, 4:] = mms_x.transform(test_x.iloc[:, 4:]).astype(np.float32)

mms_y = MinMaxScaler()
mms_y.fit(train_y.values.reshape(-1,1))
train_y = mms_y.transform(train_y.values.reshape(-1,1))
test_y = mms_y.transform(test_y.values.reshape(-1,1))

train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)

# 前12時間分のデータを参照する
timesteps = 12

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
from tensorflow.keras.callbacks import EarlyStopping # type: ignore <=　VSCode側の警告無視

from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

neuron = 126
actfunc = "tanh"
dropout = 0.2
epochs = 300

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
early_stopping =  EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.0001)

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

# #========
# #LSTM
# #========

# start = time.perf_counter() #計測開始

# model_LSTM_64 = Sequential()
# model_LSTM_64.add(Input(shape=(timesteps, x_train.shape[2])))
# model_LSTM_64.add(LSTM(neuron, activation=actfunc, return_sequences=False))
# model_LSTM_64.add(Dropout(dropout))
# model_LSTM_64.add(Dense(1, activation="linear")) 

# model_LSTM_64.compile(loss="mean_squared_error", optimizer="adam")
# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     patience=3,
#     restore_best_weights=True
# )
# history_LSTM_64 = model_LSTM_64.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

# end = time.perf_counter() #計測終了
# print(end-start)

# plt.figure(figsize=(15,8))
# plt.plot(history_LSTM_64.history['loss'], label='Train Loss')
# plt.plot(history_LSTM_64.history['val_loss'], label='valid Loss')
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()

#========
#GRU
#========

y_test_pred_GRU_64 = model_GRU_64.predict(x_test)
y_test_pred_GRU_64 = mms_y.inverse_transform(y_test_pred_GRU_64)
y_t = mms_y.inverse_transform(y_test)

# 予測対象時刻の昼夜フラグを取得
daylight_test = df_all["daylight"].iloc[train_size + timesteps:].to_numpy()

# 夜間の予測値を0にする
y_test_pred_GRU_64[daylight_test == 0] = 0

# マイナスの予測値も0に補正
y_test_pred_GRU_64 = np.maximum(y_test_pred_GRU_64, 0)

# 予測対象の日時
test_dates = df_all.index[train_size + timesteps:]

# 予測結果を日時付きのSeriesに変換
pred_GRU = pd.Series(
    y_test_pred_GRU_64.flatten(),
    index=test_dates,
    name="pred_GRU"
)

y_test_pred_GRU_64 = pd.DataFrame(
    y_test_pred_GRU_64,
    index=test_dates
)

y_t = pd.DataFrame(
    y_t,
    index=test_dates
)

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

# #========
# #LSTM
# #========

# y_test_pred_LSTM_64 = model_LSTM_64.predict(x_test)
# y_test_pred_LSTM_64 = mms_y.inverse_transform(y_test_pred_LSTM_64)
# y_t = mms_y.inverse_transform(y_test)

# LSTM_RMSE_64 = np.sqrt(mean_squared_error(y_t, y_test_pred_LSTM_64))
# LSTM_MAE_64 = mean_absolute_error(y_t, y_test_pred_LSTM_64)
# LSTM_MAPE_64 = mean_absolute_percentage_error(y_t, y_test_pred_LSTM_64)

# # 指標出力
# print('RMSE:')
# print(LSTM_RMSE_64)
# print('MAE:')
# print(LSTM_MAE_64) 
# print('MAPE:')
# print(LSTM_MAPE_64) 

# #テストデータ(青)と予測(黄色)を描画
# plt.figure(figsize=(30,8))
# plt.plot(y_t, c="b")
# plt.plot(y_test_pred_LSTM_64, c="y")
# plt.show()

# 後ろから2週間分(24*7*2)のデータだけ描画
plt.figure(figsize=(30,8))
plt.plot(y_t[-336:], c="b")
# plt.plot(y_test_pred_GRU_64[-336:], c="y")
plt.plot(y_test_pred_GRU_64[-336:], c="r")
plt.show()

pred_GRU.to_csv("gen_pred.csv")

plt.scatter(y_test_pred_GRU_64, y_t)
plt.xlabel('y_t')
plt.ylabel('predictions')
plt.title('y_t vs predictions')
plt.plot([min(y_t), max(y_t)], [min(y_t), max(y_t)], color='red')  # 45度線
plt.show()