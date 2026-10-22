import pandas as pd
import datetime

def getEstedPowerGeneration(yyyy, MM, dd, hh, mm, df):
    if(mm == 0 or mm == 30):
        dt = datetime.datetime(yyyy, MM, dd, hh, mm, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
        #print(dt)
        return df[df["datetime"] == dt]
        # else: return "does not exist"
    else: return "minutes error"

df = pd.read_csv("gen_pred.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# print(df.head)
# print(type(df["datetime"].iloc[0]))

print(getEstedPowerGeneration(2026, 7, 5, 6, 0, df)["pred_GRU"])