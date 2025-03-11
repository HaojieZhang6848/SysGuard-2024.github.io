import pandas as pd
import os
import json

DEVICE_LOG_INPUT = os.environ.get('DEVICE_LOG_INPUT', 'Dataset/Smart Home/Device Log')
SORTED_COMBINED_LOG = os.environ.get('SORTED_COMBINED_LOG', 'data/01_device_sorted_combined_log.xlsx')
DEVICE_STATE_HISTORY = os.environ.get('DEVICE_STATE_HISTORY', 'data/02_device_state_history.xlsx')
TIME_SERIES_WINDOWS = os.environ.get('TIME_SERIES_WINDOWS', 'data/03_time_series_windows.xlsx')
KEEP_TIMESTAMP = os.getenv('KEEP_TIMESTAMP') is not None

TAU = int(os.getenv('TAU', 5))

actuator_types = [
    'AC',
    'AirPurifier',
    'BathHeater',
    'CookerHood',
    'Curtain',
    'Door',
    'Fridge',
    'GasStove',
    'Heater',
    'Humidifier',
    'Light',
    'MicrowaveOven',
    'Printer',
    'Projector',
    'Speaker',
    'TV',
    'TowelDryer',
    'WashingMachine',
    'WaterDispenser',
    'WaterHeater',
    'Window',
]

devices = []

with open('Dataset/Smart Home/device_value_mapping.json', 'r') as file:
    device_value_mapping = json.load(file)


def get_device_type(device: str) -> str:
    for device_type, _ in device_value_mapping.items():
        if device_type in device:
            return device_type
    return 'Unknown'


# Traverse all .xlsx files in the directory (i.e Device Log)
all_data = []
excluded_devices = ['Weather', 'HumanCount', 'HumanState']
device_log_path = DEVICE_LOG_INPUT
for root, _, files in os.walk(device_log_path):
    for file in files:
        if file.endswith('.xlsx') and not any(device in file for device in excluded_devices):
            file_path = os.path.join(root, file)
            data = pd.read_excel(file_path)
            all_data.append(data)

            device_name_split = file.split('.')
            device_name = device_name_split[0] + '_' + device_name_split[1]
            devices.append(device_name)

# Merge all data into a single DataFrame
os.makedirs('data', exist_ok=True)
time_series_data = pd.concat(all_data, ignore_index=True)
time_series_data.sort_values(by='Timestamp', inplace=True)
time_series_data.to_excel(SORTED_COMBINED_LOG, index=False)
print(f"Successfully exported to {SORTED_COMBINED_LOG}")

raw_df = pd.read_excel(SORTED_COMBINED_LOG)
columns = ['Timestamp'] + devices
time_series_data = pd.DataFrame(columns=columns)

# group by timestamp
raw_df_grouped = raw_df.groupby('Timestamp')
for timestamp in sorted(raw_df['Timestamp'].unique()):
    row_data = {}
    row_data['Timestamp'] = timestamp
    for device in devices:
        device_type = get_device_type(device)
        device_name = device.replace('_', '.')
        log_group = raw_df_grouped.get_group(timestamp)
        state = log_group[log_group['stateInfo'].str.contains(device_name)]
        if not state.empty:
            payload_data = state['stateInfo'].values[0]
            value = payload_data.split(device_name + '.state: ')[1]
            row_data[device] = device_value_mapping[device_type][value]
        else:
            row_data[device] = None
    time_series_data = pd.concat([time_series_data, pd.DataFrame([row_data])], ignore_index=True)

time_series_data.ffill(inplace=True)
time_series_data.fillna(0, inplace=True)

time_series_data.to_excel(DEVICE_STATE_HISTORY, index=False)
print(f"Successfully exported to {DEVICE_STATE_HISTORY}")

# Generate time series windows
def calc_time_series_windows(time_series: pd.DataFrame) -> pd.DataFrame:
    if not KEEP_TIMESTAMP:
        # drop timestamp and seq columns, unuseful for structure and parameter learning
        time_series = time_series.drop(columns=['Timestamp'])

    time_series_shifted_i = []
    time_series_shifted_i.append(time_series)

    for i in range(1, TAU+1):
        time_series_shifted_i.append(time_series.shift(-i).dropna().reset_index(drop=True))

    for i in range(len(time_series_shifted_i)):
        time_series_shifted_i[i].columns = [f"{col}_{i}" for col in time_series_shifted_i[i].columns]
        if i - TAU >= 0:
            time_series_shifted_i[i] = time_series_shifted_i[i]
        else:
            time_series_shifted_i[i] = time_series_shifted_i[i].iloc[:i - TAU]

    time_series_windows = pd.concat(time_series_shifted_i, axis=1)
    time_series_windows.to_excel(TIME_SERIES_WINDOWS, index=False)

    return time_series_windows

calc_time_series_windows(time_series_data)
print(f"Successfully exported to {TIME_SERIES_WINDOWS}")