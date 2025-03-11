import json


def get_device_type(device_name):
    _, device_name = device_name.split('_')
    return device_name[:-3]


def get_interested_device_names():
    with open('data/device_names.json', 'r') as f:
        device_names = json.load(f)
        interested_device_types = [
            'AC',
            'AirPurifier',
            'BathHeater',
            'CookerHood',
            'Curtain',
            'GasStove',
            'Heater',
            'Humidifier',
            'Light',
            'TV',
            'WashingMachine',
        ]
        device_names = [d for d in device_names if get_device_type(d) in interested_device_types]
    return device_names


device_names = get_interested_device_names()
device_id_to_name = {i: d_name for i, d_name in enumerate(device_names)}
device_name_to_id = {d_name: i for i, d_name in enumerate(device_names)}

state_extremas = {
    'AirQuality': [0, 3],
    'Brightness': [0, 3],
    'COLevel': [0, 2],
    'EnergyConsumption': [0, 2],
    'Humidity': [0, 4],
    'Noise': [0, 3],
    'Temperature': [0, 4],
    'UltravioletLevel': [0, 4],
}
