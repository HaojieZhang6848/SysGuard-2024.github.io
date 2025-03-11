import json
import random
import bnlearn as bn
import os

import pandas as pd
from other_utils import get_interested_device_names, get_device_type
from lstm2_utils import calc_lstm2_in_feature_names, calc_save_device_lstm2_in_feature_names
from pgm_utils import cleanise_dataset, calc_valid_columns, calc_influenced_nodes, calc_influenced_nodes_value
from collections import defaultdict
import json
import numpy as np

np.random.seed(42)
random.seed(42)

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

TAU = int(os.getenv('TAU', 2))
DAG = bn.load('data/DAG-b.pkl')

device_names = get_interested_device_names()
device_name_to_id = {device_name: i for i, device_name in enumerate(device_names)}

device_lstm2_in_feature_names = calc_save_device_lstm2_in_feature_names(DAG, device_names)

n_in_features = max([len(device_lstm2_in_feature_names[device_name]['in']) for device_name in device_names])  # 9
print(f'n_in_features: {n_in_features}')


def preprocess_train_data():
    time_series_windows_test = pd.read_excel(f'data/03_time_series_windows_train2_{TAU}.xlsx')
    valid_columns = calc_valid_columns(time_series_windows_test, DAG)
    time_series_windows_test = cleanise_dataset(time_series_windows_test, DAG, valid_columns)

    samples_each_device = 1000

    # (batch_size, seq_len, n_features)
    lstm2_train_x = np.zeros((samples_each_device * len(device_names), TAU + 1, n_in_features))
    lstm2_train_labels = np.zeros(samples_each_device * len(device_names))  # 0: Normal, 1: Abnormal

    print('Start inference...')
    for i, d_name in enumerate(device_names):
        print(f'Inferencing {d_name}...')
        # influenced_nodes = calc_influenced_nodes(d_name, DAG)
        # last_influenced_nodes = [node for node in influenced_nodes if node.endswith(f'_{TAU}')]

        # inf_pred = calc_influenced_nodes_value(DAG, time_series_windows_test, valid_columns, influenced_nodes)
        # inf_pred = inf_pred[last_influenced_nodes]

        normal_samples = samples_each_device // 2

        for j in range(samples_each_device):
            sample_idx = i * samples_each_device + j
            for feature_idx, feature_name in enumerate(device_lstm2_in_feature_names[d_name]['in']):
                if feature_name == 'device_id':
                    lstm2_train_x[sample_idx, :, feature_idx] = device_name_to_id[d_name]
                    continue
                if feature_name.endswith('_PgmPred'):
                    f_name = feature_name.replace('_PgmPred', '')
                    f_name = f'{f_name}_{TAU}'
                    lstm2_train_x[sample_idx, :, feature_idx] = 0
                    continue
                for seq_idx in range(TAU+1):
                    f_name = f'{feature_name}_{seq_idx}'
                    true_val = time_series_windows_test[f_name].iloc[j]
                    state_type = f_name.split('_')[1]
                    if j >= normal_samples and state_type in state_extremas.keys():
                        min_val, max_val = state_extremas[state_type]
                        dist_to_min = abs(true_val - min_val)
                        dist_to_max = abs(true_val - max_val)
                        lstm2_train_x[sample_idx, seq_idx, feature_idx] = min_val if dist_to_min > dist_to_max else max_val
                    else:
                        lstm2_train_x[sample_idx, seq_idx, feature_idx] = true_val
            
            lstm2_train_labels[sample_idx] = 1 if j >= normal_samples else 0
            
        print(f'Inferencing {d_name} done.')

    # shuffle the data
    indices = np.arange(len(lstm2_train_x))
    np.random.shuffle(indices)
    lstm2_train_x = lstm2_train_x[indices]
    lstm2_train_labels = lstm2_train_labels[indices]

    print('Inference done.')
    print('Start saving...')
    np.save('data-dc/lstm2_train_x.npy', lstm2_train_x)  # (batch_size, seq_len, n_in_features)
    np.save('data-dc/lstm2_train_labels.npy', lstm2_train_labels)  # (batch_size,)
    print('Saving done.')


def preprocess_test_data():
    # 加载数据集
    dataset = pd.read_csv(f'data-dc/04_fault_detection_dataset_{TAU}.csv')
    dataset = dataset.reset_index(drop=True)

    # Predict based on the DAG
    input_columns = calc_valid_columns(dataset, DAG)

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

    # cleanise dataset
    dataset = cleanise_dataset(dataset, DAG, input_columns)

    def device_filter(row):
        return get_device_type(row['Device']) in interested_device_types
    dataset = dataset[dataset.apply(device_filter, axis=1)]

    # group dataset by device name
    dataset_grouped = dataset.groupby('Device')

    lstm2_test_x = np.zeros((len(dataset), TAU + 1, n_in_features))
    lstm2_test_labels = np.zeros(len(dataset))  # 0: Normal, 1: Abnormal
    lstm2_test_on_off = np.zeros(len(dataset))  # 0: Off, 1: On

    sample_idx = 0

    for d_name, group in dataset_grouped:
        device_type = get_device_type(d_name)
        if device_type not in interested_device_types:
            continue
        print(f'Inferencing {d_name}...')

        # influenced_nodes = calc_influenced_nodes(d_name, DAG)
        # last_influenced_nodes = [node for node in influenced_nodes if node.endswith(f'_{TAU}')]
        # inf_pred = calc_influenced_nodes_value(DAG, group, input_columns, influenced_nodes)
        # inf_pred = inf_pred[last_influenced_nodes]

        n_time_series = len(group)
        for j in range(n_time_series):
            for feature_idx, feature_name in enumerate(device_lstm2_in_feature_names[d_name]['in']):
                if feature_name == 'device_id':
                    lstm2_test_x[sample_idx, :, feature_idx] = device_name_to_id[d_name]
                    continue
                if feature_name.endswith('_PgmPred'):
                    f_name = feature_name.replace('_PgmPred', '')
                    f_name = f'{f_name}_{TAU}'
                    lstm2_test_x[sample_idx, :, feature_idx] = 0
                    continue
                for seq_idx in range(TAU + 1):
                    f_name = f'{feature_name}_{seq_idx}'
                    lstm2_test_x[sample_idx, seq_idx, feature_idx] = group[f_name].iloc[j]

            label = group['Label'].iloc[j]
            lstm2_test_labels[sample_idx] = 1 if label == 'Abnormal' else 0
            onoff = group[f'{d_name}_0'].iloc[j]
            lstm2_test_on_off[sample_idx] = onoff
            sample_idx += 1

        print(f'Inferencing {d_name} done.')

    assert sample_idx == len(dataset)
    np.save('data-dc/lstm2_test_x.npy', lstm2_test_x)  # (batch_size, seq_len, n_in_features)
    np.save('data-dc/lstm2_test_labels.npy', lstm2_test_labels)  # (batch_size,)
    np.save('data-dc/lstm2_test_on_off.npy', lstm2_test_on_off)  # (batch_size,)
    print('Saving done.')


if __name__ == '__main__':
    print('========Preprocessing test data...========')
    preprocess_test_data()
    print('========Preprocessing train data...========')
    preprocess_train_data()
    print('========Done.========')
