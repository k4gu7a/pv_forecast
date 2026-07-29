import requests
import os
from dotenv import load_dotenv
import pandas as pd
import time

start = time.perf_counter()

load_dotenv()

API_KEY = os.getenv("API_KEY")
MY_SYSTEM_ID = os.getenv("MY_SYSTEM_ID")
TARGET_SYSTEM_ID = os.getenv("TARGET_SYSTEM_ID")

TARGET_DATE_FROM = "20260326"
TARGET_DATE_TO = "20260729"


url = "https://pvoutput.org/service/r2/getstatus.jsp"

headers = {
    "X-Pvoutput-Apikey": API_KEY,
    "X-Pvoutput-SystemId": MY_SYSTEM_ID,

    "X-Rate-Limit": "1",
}

dates = pd.date_range(
    start=TARGET_DATE_FROM,
    end=TARGET_DATE_TO,
    freq="D"
)

def download_from_api(d):
    
    params = {
        "d": d,
        "h": 1,
        "limit": 288,
        "asc": 1,
        "sid1": TARGET_SYSTEM_ID,
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        print("HTTP status:", response.status_code)

        print(
            "Rate Limit:",
            response.headers.get("X-Rate-Limit-Limit")
        )
        print(
            "Remaining:",
            response.headers.get("X-Rate-Limit-Remaining")
        )

        print(
            "Reset:",
            pd.to_datetime(
                int(response.headers.get("X-Rate-Limit-Reset")),
                unit="s",
                utc=True).tz_convert("Asia/Tokyo")
        )

        #print(response.text[:1000])

        response.raise_for_status() #http status ≠ 200 例外処理

        records = [
            record.strip()
            for record in response.text.strip().split(";")
            if record.strip()
        ]

        print("\nsuccess")
        print("records:", len(records))

    except requests.exceptions.HTTPError:
        print("\nhttp error occurred:", response.text)
        return None

    except requests.exceptions.RequestException as e:
        print("\nconnection error occurred:", e)
        return None


    columns = [     #取得順
        "date",
        "time",
        "energy_generation_wh",
        "energy_efficiency_kwh_per_kw",
        "instantaneous_power_w",
        "average_power_w",
        "normalised_output_kw_per_kw",
        "energy_consumption_wh",
        "power_consumption_w",
        "temperature_c",
        "voltage_v",
    ]

    rows = [record.split(",") for record in records]

    for i, row in enumerate(rows):
        if len(row) != len(columns):
            raise ValueError(
                f"Invalid number of rows: line {i+1}, {len(row)} columns (Expected: {len(columns)})"
            )

    df = pd.DataFrame(rows, columns=columns)

    for col in columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y%m%d %H:%M"
    )

    df = df.set_index("datetime").asfreq("5min")

    df = df[
        [
            "energy_generation_wh",
            "energy_efficiency_kwh_per_kw",
            "instantaneous_power_w",
            "average_power_w",
            "normalised_output_kw_per_kw",
            "temperature_c",
            "voltage_v",
            "energy_consumption_wh",
            "power_consumption_w",
        ]
    ]

    return df

dfs = []

for date in dates:

    date_str = date.strftime("%Y%m%d")

    df_raw = download_from_api(date_str)

    if df_raw is None:
        print(f"{date_str}: skipped")
        continue

    raw_filename = f"{TARGET_SYSTEM_ID}_raw_{date_str}.csv"
    processed_filename = f"{TARGET_SYSTEM_ID}_pro_{TARGET_DATE_FROM}-{TARGET_DATE_TO}.csv"

    df_raw.to_csv(
        "data/power_generation/pvoutput/112207/raw/" + raw_filename,
        index=True,
        encoding="utf-8-sig"
    )

    # print(df.head())
    # print(df.iloc[94])
    # print(df["energy_consumption_wh"].dtype)

    print(f"\nsaved rawfile as{raw_filename}")

    df_processed = df_raw[["average_power_w"]].interpolate(method="linear", limit=3)
    df_processed.iloc[0] = 0

    #print(df_processed)

    df_pv_30min = df_processed.resample("30min").sum() / 12

    dfs.append(df_pv_30min)

df_all = pd.concat(dfs)
df_all.to_csv(
    "data/power_generation/pvoutput/112207/processed/" + processed_filename,
    index=True,
    encoding="utf-8-sig"
)

print(f"\nsaved processedfile as{processed_filename}")

end = time.perf_counter()
print('{:.2f}'.format((end-start)/60))