"""10:00時点の翌日30分PV予測: 39時間予報の理想値を使う疑似実験。

注意: 入力気象CSVは有効時刻ごとの値で、発表時刻がない。未来の気象値を
「当日06:00初期値の予報」とみなす *完全予報の代理実験* であり、実運用の
予報精度を含まない。実際の06:00予報を評価するには発表時刻別アーカイブが必要。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xarray as xr
from astral import Observer
from astral.sun import sun
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========= ファイル指定 =========
WEATHER_CSV = Path('data/all.csv')   # 存在すれば優先: datetime + 気象列など
WEATHER_NC = Path('data/msm-s_auto/all.nc')  # CSVが無ければこちらから読み込む
PV_CSV = Path('data/power_generation/pvoutput/112207/processed/112207_pro_20260326-20260729.csv')
OUTPUT_DIR = Path('results_lgbm_pseudo')
OUTPUT_DIR.mkdir(exist_ok=True)
TZ = 'Asia/Tokyo'
LAT, LON = 35.7999995, 139.0625
ISSUE_HOUR = 6                   # 気象予報の初期時刻 (06:00 JST)
DECISION_HOUR = 10               # 予測を行う時刻 (10:00 JST)
HORIZON_HOURS = 39               # 06時初期値からの予報範囲
TRAIN_RATIO = 0.8                # 予測実行日単位で時系列分割

# ========= 読み込み =========
def read_time_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace(
        "**", "", regex=False
    )

    stamps = pd.to_datetime(df["datetime"])

    if stamps.dt.tz is None:
        # タイムゾーンなし → JSTとして解釈
        stamps = stamps.dt.tz_localize("Asia/Tokyo")
    else:
        # タイムゾーンあり → JSTに変換
        stamps = stamps.dt.tz_convert("Asia/Tokyo")

    df["datetime"] = stamps

    return df.set_index("datetime").sort_index()

if WEATHER_CSV.exists():
    weather_raw = read_time_csv(WEATHER_CSV)
elif WEATHER_NC.exists():
    with xr.open_dataset(WEATHER_NC) as ds:
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            ds = ds.sel(latitude=LAT, longitude=LON, method='nearest')
        # この疑似実験が扱えるのは、有効時刻(time)を添字とする気象値。
        # 初期値時刻×予報対象時刻が残る多次元アーカイブは別途処理が必要。
        if 'time' not in ds.coords or 'time' not in ds.dims:
            raise ValueError('ncに一次元の time 座標がありません。予報初期時刻別データなら別の読み方が必要です')
        weather_raw = ds.to_dataframe().reset_index()
    weather_raw['datetime'] = pd.to_datetime(weather_raw.pop('time'), utc=True).dt.tz_convert(TZ)
    weather_raw = weather_raw.set_index('datetime').sort_index()
    if weather_raw.index.has_duplicates:
        raise ValueError('ncに同じ有効時刻の複数予報があります。初期時刻ごとの選択が必要です')
else:
    raise FileNotFoundError(f'{WEATHER_CSV} も {WEATHER_NC} も見つかりません')
pv_raw = read_time_csv(PV_CSV)
if 'average_power_w' not in pv_raw:
    raise ValueError('発電データに average_power_w 列が必要です')

# 1時間ごとの気象: 時刻Tの値を [T, T+1h) に対応する予報値として使用
weather_cols = [
    'UGRD_10maboveground', 'VGRD_10maboveground',
    'TMP_1D5maboveground', 'RH_1D5maboveground',
    'LCDC_surface', 'MCDC_surface', 'HCDC_surface', 'TCDC_surface',
    'APCP_surface', 'DSWRF_surface'
]
missing = [c for c in weather_cols if c not in weather_raw.columns]
if missing:
    raise ValueError(f'気象CSVに列がありません: {missing}。実際の列名を確認してください: {list(weather_raw.columns)}')
weather = weather_raw[weather_cols].apply(pd.to_numeric, errors='coerce').resample('1h').mean()

# 発電値が「5分間平均の電力 [W]」であることを想定。
# 30分平均電力 [W] を学習し、最後に kWh/30分 へ換算する。
pv = pd.to_numeric(pv_raw['average_power_w'], errors='coerce').resample('30min').mean()
pv.name = 'power_w'

# 同一日06時初期値+39時間まで、翌日全コマをカバー可能（翌日21時まで）
# 行単位の欠損を除外し、比較の対象時刻を揃える。
observer = Observer(latitude=LAT, longitude=LON)
sun_cache = {}
def solar_features(ts):
    d = ts.date()
    if d not in sun_cache:
        sun_cache[d] = sun(observer, date=d, tzinfo=TZ)
    sunrise, sunset = sun_cache[d]['sunrise'], sun_cache[d]['sunset']
    midpoint = ts + pd.Timedelta(minutes=15)
    daylight = int(sunrise <= midpoint < sunset)
    return daylight, (midpoint - sunrise).total_seconds()/3600, (sunset - midpoint).total_seconds()/3600

rows = []
# 過去の実績は「予測日の10:00より前」のデータのみ使用
for date in pd.date_range(pv.index.min().normalize(), pv.index.max().normalize(), freq='D', tz=TZ):
    issue = date + pd.Timedelta(hours=ISSUE_HOUR)
    decision = date + pd.Timedelta(hours=DECISION_HOUR)
    target_day = date + pd.Timedelta(days=1)
    if not (pv.index.min() <= decision and target_day <= pv.index.max()):
        continue
    past = pv[(pv.index >= decision - pd.Timedelta(hours=24)) & (pv.index < decision)]
    if len(past) < 40 or past.isna().any():  # 24hのうち十分な履歴がある予測日のみ
        continue
    prev_day_power = past.sum() * 0.5 / 1000.0   # 24h kWh
    recent_3h_power = past.tail(6).mean()       # W
    for step in range(48):
        t = target_day + pd.Timedelta(minutes=30*step)
        midpoint = t + pd.Timedelta(minutes=15)
        # この時刻の気象予報が06時初期値の39時間範囲内にあるか
        lead_h = (t - issue).total_seconds()/3600
        if not (0 <= lead_h < HORIZON_HOURS):
            continue
        h = t.floor('h')
        if h not in weather.index or t not in pv.index:
            continue
        wx = weather.loc[h]
        if wx.isna().any() or pd.isna(pv.loc[t]):
            continue
        daylight, hrs_after_rise, hrs_to_set = solar_features(t)
        row = wx.to_dict()
        row.update({
            'decision_time': decision, 'target_time': t,
            'actual_w': float(pv.loc[t]),
            'lead_h': lead_h,
            'hour_sin': np.sin(2*np.pi*(midpoint.hour + midpoint.minute/60)/24),
            'hour_cos': np.cos(2*np.pi*(midpoint.hour + midpoint.minute/60)/24),
            'dayofyear_sin': np.sin(2*np.pi*midpoint.dayofyear/365.25),
            'dayofyear_cos': np.cos(2*np.pi*midpoint.dayofyear/365.25),
            'daylight': daylight,
            'hours_after_sunrise': hrs_after_rise,
            'hours_before_sunset': hrs_to_set,
            'past_24h_kwh': prev_day_power,
            'recent_3h_mean_w': recent_3h_power,
            'wind_speed': np.hypot(row['UGRD_10maboveground'], row['VGRD_10maboveground']),
        })
        rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError('学習行が0件です。データ期間・日時・欠損を確認してください')
df = df.sort_values(['decision_time','target_time']).reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ('decision_time','target_time','actual_w')]

# 日付単位に時系列分割（同じ10時発表分が学習・テストの両方に入らない）
issue_dates = pd.Index(df['decision_time'].drop_duplicates())
cut = max(1, min(len(issue_dates)-1, int(len(issue_dates)*TRAIN_RATIO)))
if len(issue_dates) < 3:
    raise RuntimeError('少なくとも3つの予測実行日のデータが必要です')
train = df[df['decision_time'].isin(issue_dates[:cut])].copy()
test = df[df['decision_time'].isin(issue_dates[cut:])].copy()
X_train, y_train = train[feature_cols], train['actual_w']
X_test, y_test = test[feature_cols], test['actual_w']

# ========= LightGBM =========
model = lgb.LGBMRegressor(
    n_estimators=400, learning_rate=0.03, num_leaves=15,
    min_child_samples=20, random_state=42, verbosity=-1,
)
model.fit(X_train, y_train)
test['pred_lgbm_w'] = model.predict(X_test)
test['pred_lgbm_w'] = np.maximum(test['pred_lgbm_w'], 0)
test.loc[test['daylight'] == 0, 'pred_lgbm_w'] = 0

# ========= 比較用: 日射量のみに基づく単純な線形回帰 =========
# 係数は学習データでのみ推定する（テストデータを使わない）。
# 切片なし、日射量0なら予測発電量0とする。
s = np.maximum(train['DSWRF_surface'].to_numpy(), 0)
coef = float(np.dot(s, y_train.to_numpy()) / np.dot(s, s)) if np.dot(s, s) > 0 else 0.0
test['pred_radiation_w'] = np.maximum(test['DSWRF_surface'].to_numpy(), 0) * coef
test.loc[test['daylight'] == 0, 'pred_radiation_w'] = 0

# ========= 評価: 30分ごとの発電電力量[kWh]に換算 =========
for col in ['actual_w', 'pred_lgbm_w', 'pred_radiation_w']:
    test[col.replace('_w', '_kwh')] = test[col] * 0.5 / 1000.0

def metrics(actual, pred):
    return {'MAE_kWh': mean_absolute_error(actual, pred),
            'RMSE_kWh': np.sqrt(mean_squared_error(actual, pred))}

result = pd.DataFrame({
    'LightGBM': metrics(test['actual_kwh'], test['pred_lgbm_kwh']),
    'Radiation baseline': metrics(test['actual_kwh'], test['pred_radiation_kwh']),
}).T
print('=== 30分コマごとの精度 ===')
print(result.to_string())
print(f'学習: {train.decision_time.min()} 〜 {train.decision_time.max()} ({len(train)}コマ)')
print(f'テスト: {test.decision_time.min()} 〜 {test.decision_time.max()} ({len(test)}コマ)')

# 1日合計の誤差（夜間も含めて計算。翌日中の21時以降は予報範囲外だがPVはゼロと仮定しない）
daily = test.groupby('decision_time')[['actual_kwh', 'pred_lgbm_kwh', 'pred_radiation_kwh']].sum()
print('\n=== 翌日合計の精度 ===')
for name, col in [('LightGBM','pred_lgbm_kwh'), ('Radiation baseline','pred_radiation_kwh')]:
    print(name, metrics(daily['actual_kwh'], daily[col]))

print('\n=== 特徴量重要度 上位15 ===')
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(imp.head(15))

# ========= 保存と可視化 =========
test.to_csv(OUTPUT_DIR/'predictions.csv', index=False, encoding='utf-8-sig')
daily.to_csv(OUTPUT_DIR/'daily_totals.csv', encoding='utf-8-sig')
result.to_csv(OUTPUT_DIR/'metrics.csv', encoding='utf-8-sig')
imp.to_csv(OUTPUT_DIR/'feature_importance.csv', encoding='utf-8-sig')

plot_df = test.set_index('target_time').tail(48*7)
plt.figure(figsize=(15, 5))
plt.plot(plot_df.index, plot_df['actual_kwh'], label='Actual')
plt.plot(plot_df.index, plot_df['pred_lgbm_kwh'], label='LightGBM')
plt.plot(plot_df.index, plot_df['pred_radiation_kwh'], label='Radiation baseline', alpha=.75)
plt.ylabel('Generated energy [kWh / 30min]')
plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR/'prediction_last_week.png', dpi=150)
plt.show()

print('\n重要: 気象CSVには予報初期時刻がないため、これは将来の実現気象を予報の代わりにした楽観的な疑似実験です。')
print('実際の06時初期値予報の誤差を含む性能評価ではありません。')
