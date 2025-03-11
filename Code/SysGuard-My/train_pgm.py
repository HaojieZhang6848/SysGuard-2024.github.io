import bnlearn as bn
import pandas as pd
import numpy as np
import sys
import requests
import os

TAU = 2
METHOD = os.getenv('METHOD', 'a')

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

actuator_chinese = {
    'AC': '空调',
    'AirPurifier': '空气净化器',
    'BathHeater': '浴霸',
    'CookerHood': '抽油烟机',
    'Curtain': '窗帘',
    'Door': '门',
    'Fridge': '冰箱',
    'GasStove': '燃气灶',
    'Heater': '取暖器',
    'Humidifier': '加湿器',
    'Light': '灯',
    'MicrowaveOven': '微波炉',
    'Printer': '打印机',
    'Projector': '投影仪',
    'Speaker': '音响',
    'TV': '电视',
    'TowelDryer': '烘干机',
    'WashingMachine': '洗衣机',
    'WaterDispenser': '饮水机',
    'WaterHeater': '热水器',
    'Window': '窗户',
}

sensor_types = [
    'AirQuality',
    'Brightness',
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

sensor_chinese = {
    'AirQuality': '空气质量',
    'Brightness': '亮度',
    'COLevel': '一氧化碳浓度',
    'EnergyConsumption': '能耗',
    'HumanCount': '人数',
    'HumanState': '人员状态',
    'Humidity': '湿度',
    'Noise': '噪音',
    'Temperature': '温度',
    'UltravioletLevel': '紫外线强度',
    'Weather': '天气',
}

room_chinese = {
    'Balcony': '阳台',
    'Bathroom': '浴室',
    'BedroomOne': '卧室一',
    'BedroomTwo': '卧室二',
    'Cloakroom': '衣帽间',
    'Context': '外界环境',
    'Kitchen': '厨房',
    'LivingRoom': '客厅',
}


def is_sensor(name: str) -> bool:
    for st in sensor_types:
        if st in name:
            return True
    return False


def is_actuator(name: str) -> bool:
    for at in actuator_types:
        if at in name:
            return True
    return False


def prune_edge(DAG, u, v):
    DAG['model_edges'].remove((u, v))
    DAG['model'].remove_edge(u, v)
    DAG['adjmat'].loc[u, v] = False
    return DAG


def should_prune_edge(u, v):
    loc1, name1, _ = u.split('_')
    loc2, name2, _ = v.split('_')
    if is_actuator(name1):
        name1 = actuator_chinese[name1[:-3]] if name1[:-3] in actuator_chinese else name1[:-3]
        name2 = sensor_chinese[name2] if name2 in sensor_chinese else name2
        loc1 = room_chinese[loc1] if loc1 in room_chinese else loc1
        loc2 = room_chinese[loc2] if loc2 in room_chinese else loc2
        prompt = f"""
        你是一个擅长分析设备交互，或设备与环境属性交互，或环境属性与环境属性交互的专家。
        现在，请你分析一下，空间【{loc1}】的设备【{name1}】的工作状态，是否会对空间【{loc2}】的环境属性【{name2}】产生影响？
        请使用Yes或No回答，不要使用其他词语。
        """
    else:
        name1 = sensor_chinese[name1] if name1 in sensor_chinese else name1
        name2 = sensor_chinese[name2] if name2 in sensor_chinese else name2
        loc1 = room_chinese[loc1] if loc1 in room_chinese else loc1
        loc2 = room_chinese[loc2] if loc2 in room_chinese else loc2
        prompt = f"""
        你是一个擅长分析设备交互，或设备与环境属性交互，或环境属性与环境属性交互的专家。
        现在，请你分析一下，空间【{loc1}】的环境属性【{name1}】，是否会对空间【{loc2}】的环境属性【{name2}】产生影响？
        请使用Yes或No回答，不要使用其他词语。
        """

    req_body = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    }
    url = 'http://192.168.1.84:11434/api/generate'
    rsp_body = requests.post(url, json=req_body).json()['response']

    import re

    def remove_think_tags(input_string):
        # 使用正则表达式去除<think>...</think>之间的内容
        return re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
    rsp_body = remove_think_tags(rsp_body)

    if "yes" in rsp_body.lower():
        return False
    return True


def should_keep_edge(u, v):
    rules = {
        "Door": [
            "Humidity",
            "Temperature",
            "AirQuality",
            "COLevel"
        ],
        "Window": ["Humidity", "Temperature", "AirQuality", "COLevel"],
        "Light": ["Brightness"],
        "Heater": ["Temperature"],
        "CookerHood": ["AirQuality", "Noise", "COLevel"],
        "Humidifier": ["Humidity"],
        "AC": ["Temperature"],
        "Curtain": ["Brightness", "UltravioletLevel"],
        "HumanState": ["Temperature", "AirQuality"],
        "AirPurifier": ["AirQuality",  "COLevel"],
        "WaterHeater": [],
        "TowelDryer": ["Temperature"],
        "MicrowaveOven": [],
        "WashingMachine": ["Noise"],
        "BathHeater": ["Temperature"],
        "TV": ["Noise"],
        "GasStove": ["AirQuality", "COLevel"],
    }
    loc1, name1, _ = u.split('_')
    loc2, name2, _ = v.split('_')
    if is_sensor(name1) and is_sensor(name2):
        return name1 == name2 and (loc1 == 'Context' or loc1 == loc2)
    elif is_actuator(name1) and is_sensor(name2):
        device_type = name1[:-3]
        return device_type in rules and name2 in rules[device_type]
    else:
        return True


def train():
    time_series_windows = pd.read_excel('data/03_time_series_windows_train.xlsx')
    drop_columns = [f'Timestamp_{i}' for i in range(TAU+1)]
    time_series_windows = time_series_windows.drop(columns=drop_columns)
    black_list_edges = []
    for n1 in time_series_windows.columns:
        for n2 in time_series_windows.columns:
            def parse_device_name(n: str) -> str:
                return n.split('_')
            loc1, name1, seq1 = parse_device_name(n1)
            loc2, name2, seq2 = parse_device_name(n2)

            # 禁止不在同一个位置的设备之间的边，但位于Context的设备可以和其他位置的设备之间有边
            if loc1 != loc2 and loc1 != 'Context':
                black_list_edges.append((n1, n2))

            if loc1 == 'Context' and name1 != name2:
                black_list_edges.append((n1, n2))

            if is_actuator(name2):
                black_list_edges.append((n1, n2))

            seq1 = int(seq1)
            seq2 = int(seq2)
            if seq1 >= seq2 and seq1 > 0:
                black_list_edges.append((n1, n2))

    black_list_edges = list(set(black_list_edges))

    DAG = bn.structure_learning.fit(time_series_windows, methodtype='hc', black_list=black_list_edges, bw_list_method='edges')

    edges = list(DAG['model'].edges())
    for e in edges:
        loc1, name1, _ = e[0].split('_')
        loc2, name2, _ = e[1].split('_')
        print(f'Testing whether to prune edge {e}')
        if not should_keep_edge(e[0], e[1]):
            print(f'Pruning edge {e}')
            prune_edge(DAG, e[0], e[1])
        else:
            print(f'Keeping edge {e}')

    DAG = bn.parameter_learning.fit(DAG, time_series_windows)

    bn.save(DAG, 'data/DAG-rules.pkl')

    for e in DAG['model'].edges():
        print(e)


def train_with_predefined_edges():
    time_series_windows = pd.read_excel(f'data/03_time_series_windows_train1_{TAU}.xlsx')
    if METHOD == 'a':
        time_series_windows_2 = pd.read_excel(f'data/03_time_series_windows_train2_{TAU}.xlsx')
        time_series_windows = pd.concat([time_series_windows, time_series_windows_2], ignore_index=True)
    drop_columns = [f'Timestamp_{i}' for i in range(TAU+1)]
    time_series_windows = time_series_windows.drop(columns=drop_columns)
    white_list_edges = []

    device_affect_states = {
        "Door": [],
        "Window": ["Humidity", "Temperature", "AirQuality", "COLevel"],
        "Light": ["Brightness"],
        "Heater": ["Temperature"],
        "CookerHood": ["AirQuality",  "Noise", "COLevel"],
        "Humidifier": ["Humidity"],
        "AC": ["Temperature"],
        "Curtain": ["Brightness"],
        "AirPurifier": ["AirQuality",  "COLevel"],
        "WaterHeater": [],
        "TowelDryer": [],
        "MicrowaveOven": [],
        "WashingMachine": ["Noise"],
        "BathHeater": ["Temperature"],
        "TV": ["Noise"],
        "GasStove": ["AirQuality", "COLevel"]
    }

    for col1 in time_series_windows.columns:
        for col2 in time_series_windows.columns:
            if col1 == col2:
                continue

            def parse_column(n: str) -> str:
                room, name, seq = n.split('_')
                seq = int(seq)
                return room, name, seq
            room1, name1, seq1 = parse_column(col1)
            room2, name2, seq2 = parse_column(col2)
            # we only allow sensor->sensor edges and actuator->sensor edges, and disallow <any> -> actuator edges
            if is_sensor(name1) and is_sensor(name2):
                state1 = name1
                state2 = name2
                if state1 == state2 and room1 == room2 and seq1 + 1 == seq2:
                    white_list_edges.append((col1, col2))
                elif state1 == state2 and room1 == 'Context' and room2 != 'Context' and seq1 + 1 == seq2:
                    white_list_edges.append((col1, col2))

            if is_actuator(name1) and is_sensor(name2):
                device_type = name1[:-3]
                if (seq1 < seq2) and room1 == room2 and name2 in device_affect_states[device_type]:
                    white_list_edges.append((col1, col2))

    print(f'len(white_list_edges): {len(white_list_edges)}')
    print('Start building DAG with white_list_edges')
    DAG = bn.make_DAG(white_list_edges)
    print('Start parameter learning')
    DAG = bn.parameter_learning.fit(DAG, time_series_windows)

    bn.save(DAG, f'data/DAG-{METHOD}.pkl')

    for e in DAG['model'].edges():
        print(e)


if __name__ == '__main__':
    train_with_predefined_edges()
