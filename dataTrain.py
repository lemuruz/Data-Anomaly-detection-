import pandas as pd
data = pd.read_csv('sensor.csv')
data["machine_status"] = data["machine_status"].replace({
    'NORMAL': 0,
    'BROKEN': 1,
    'RECOVERING': 2
})
data = data.dropna(subset=['machine_status'])

sensor_cols = [col for col in data.columns if col.startswith("sensor_")]
data[sensor_cols] = data[sensor_cols].ffill().bfill()
data[sensor_cols] = (data[sensor_cols] - data[sensor_cols].mean()) / data[sensor_cols].std()

# for col in data.columns:
#     if col.startswith("sensor_"):
#         data[col] = data[col].fillna(data[col].mean())
#         std = data[col].std()
#         if std != 0:
#             data[col] = (data[col] - data[col].mean()) / std
print(data.head())