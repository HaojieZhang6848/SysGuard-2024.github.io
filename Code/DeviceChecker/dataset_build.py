import pandas as pd
import json

TAU = 2

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

sensor_types = [
    'AirQuality',
    'Brightness'
    'COLevel',
    'EnergyConsumption',
    'HumanCount',
    'HumanState',
    'Humidity',
    'Noise',
    'Temperature',
    'UltravioletLevel',
    'Weather',
]


def filter_samples(samples: pd.DataFrame, device_name):
    return samples
    # columns = [f'{device_name}_{TAU}' for TAU in range(TAU)]
    # return samples[samples.apply(lambda row: all(row[columns[0]] == row[col] for col in columns), axis=1)]

def build_dataset(time_series_windows_fault: pd.DataFrame, time_series_windows_normal: pd.DataFrame):
    with open('data-dc/home_device_exec_times.json', 'r') as f:
        device_exec_times = json.load(f)

    all_dataset = pd.DataFrame()

    for device_name in device_exec_times:
        fault_start = device_exec_times[device_name]['fault']['start']
        fault_end = device_exec_times[device_name]['fault']['end']
        
        device_abnormal_samples = time_series_windows_fault[time_series_windows_fault['Timestamp_0'] > fault_start]
        device_abnormal_samples = device_abnormal_samples[device_abnormal_samples[f'Timestamp_{TAU}'] < fault_end]

        device_abnormal_samples['Label'] = 'Abnormal'
        device_abnormal_samples['Device'] = device_name

        n_fault_samples = len(device_abnormal_samples)
        device_normal_samples = time_series_windows_normal.sample(n=n_fault_samples)
        device_normal_samples['Label'] = 'Normal'
        device_normal_samples['Device'] = device_name
        
        device_abnormal_samples = filter_samples(device_abnormal_samples, device_name)
        device_normal_samples = filter_samples(device_normal_samples, device_name)

        all_dataset = pd.concat([all_dataset, device_abnormal_samples, device_normal_samples], ignore_index=True)

        print(f'finished {device_name}, found {len(device_abnormal_samples)} samples, from {fault_start} to {fault_end}')

    all_dataset.to_csv(f'data-dc/04_fault_detection_dataset_{TAU}.csv', index=False)


def build_device_names():
    with open('data-dc/home_device_exec_times_full.json', 'r') as f:
        device_exec_times = json.load(f)
    device_names = []
    for k in device_exec_times:
        k = k.replace('-', '_')
        device_names.append(k)
    with open('data/device_names.json', 'w') as f:
        json.dump(device_names, f)

if __name__ == '__main__':
    # build_device_names()
    time_series_windows_fault = pd.read_excel(f'data-dc/03_time_series_windows_{TAU}.xlsx')
    time_series_windows_normal = pd.read_excel(f'data/03_time_series_windows_test_{TAU}.xlsx')
    build_dataset(time_series_windows_fault, time_series_windows_normal)
