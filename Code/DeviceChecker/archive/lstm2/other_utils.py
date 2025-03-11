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
