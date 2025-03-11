import random
import bnlearn as bn
import os

import pandas as pd
from device_utils import get_interested_device_names, get_device_type
from lstm_utils import calc_save_device_lstm_inout_feature_names
from pgm_utils import cleanise_dataset, calc_valid_columns, calc_influenced_nodes, calc_influenced_nodes_value
import json
import numpy as np

np.random.seed(42)
random.seed(42)

TAU = int(os.getenv('TAU', 2))
DAG = bn.load('data/DAG-b.pkl')

device_names = get_interested_device_names()
device_name_to_id = {device_name: i for i, device_name in enumerate(device_names)}

device_lstm_inout_feature_names = calc_save_device_lstm_inout_feature_names(DAG, device_names)

n_in_features = max([len(device_lstm_inout_feature_names[device_name]['in']) for device_name in device_names])  # 9
n_out_features = max([len(device_lstm_inout_feature_names[device_name]['out']) for device_name in device_names])  # 2
print(f'n_in_features: {n_in_features}, n_out_features: {n_out_features}')


def preprocess_train_data():
    time_series_windows_test = pd.read_excel(f'data/03_time_series_windows_train2_{TAU}.xlsx')
    valid_columns = calc_valid_columns(time_series_windows_test, DAG)
    time_series_windows_test = cleanise_dataset(time_series_windows_test, DAG, valid_columns)

    n_time_series = len(time_series_windows_test)

    # (batch_size, seq_len, n_features)
    lstm_train_x = np.zeros((n_time_series * len(device_names), TAU, n_in_features))
    lstm_train_y_diff = np.zeros((n_time_series * len(device_names), n_out_features))

    print('Start inference...')
    for i, d_name in enumerate(device_names):
        print(f'Inferencing {d_name}...')
        influenced_nodes = calc_influenced_nodes(d_name, DAG)
        last_influenced_nodes = [node for node in influenced_nodes if node.endswith(f'_{TAU}')]

        inf_pred = calc_influenced_nodes_value(DAG, time_series_windows_test, valid_columns, influenced_nodes)
        inf_pred = inf_pred[last_influenced_nodes]

        for j in range(n_time_series):
            sample_idx = i * n_time_series + j
            for feature_idx, feature_name in enumerate(device_lstm_inout_feature_names[d_name]['in']):
                if feature_name == 'device_id':
                    lstm_train_x[sample_idx, :, feature_idx] = device_name_to_id[d_name]
                    continue
                if feature_name.endswith('_PgmPred'):
                    f_name = feature_name.replace('_PgmPred', '')
                    f_name = f'{f_name}_{TAU}'
                    lstm_train_x[sample_idx, :, feature_idx] = inf_pred[f_name].iloc[j]
                    continue
                for seq_idx in range(TAU):
                    f_name = f'{feature_name}_{seq_idx}'
                    lstm_train_x[sample_idx, seq_idx, feature_idx] = time_series_windows_test[f_name].iloc[j]

            for feature_idx, feature_name in enumerate(device_lstm_inout_feature_names[d_name]['out']):
                f_name = f'{feature_name}_{TAU}'
                # let lstm learn the residual between the actual value and the PGM prediction
                lstm_train_y_diff[sample_idx, feature_idx] = time_series_windows_test[f_name].iloc[j] - inf_pred[f_name].iloc[j]
        print(f'Inferencing {d_name} done.')

    # shuffle the data
    indices = np.arange(len(lstm_train_x))
    np.random.shuffle(indices)
    lstm_train_x = lstm_train_x[indices]
    lstm_train_y_diff = lstm_train_y_diff[indices]

    print('Inference done.')
    print('Start saving...')
    np.save('data-dc/lstm_train_x.npy', lstm_train_x)  # (batch_size, seq_len, n_in_features)
    np.save('data-dc/lstm_train_y_diff.npy', lstm_train_y_diff)  # (batch_size, n_out_features)
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

    lstm_test_x = np.zeros((len(dataset), TAU, n_in_features))
    lstm_test_y_pgm = np.zeros((len(dataset), n_out_features))
    lstm_test_y_actual = np.zeros((len(dataset), n_out_features))
    lstm_test_labels = np.zeros(len(dataset))  # 0: Normal, 1: Abnormal
    lstm_test_on_off = np.zeros(len(dataset))  # 0: Off, 1: On

    sample_idx = 0

    for d_name, group in dataset_grouped:
        device_type = get_device_type(d_name)
        if device_type not in interested_device_types:
            continue
        print(f'Inferencing {d_name}...')

        influenced_nodes = calc_influenced_nodes(d_name, DAG)
        last_influenced_nodes = [node for node in influenced_nodes if node.endswith(f'_{TAU}')]
        inf_pred = calc_influenced_nodes_value(DAG, group, input_columns, influenced_nodes)
        inf_pred = inf_pred[last_influenced_nodes]

        n_time_series = len(group)
        for j in range(n_time_series):
            for feature_idx, feature_name in enumerate(device_lstm_inout_feature_names[d_name]['in']):
                if feature_name == 'device_id':
                    lstm_test_x[sample_idx, :, feature_idx] = device_name_to_id[d_name]
                    continue
                if feature_name.endswith('_PgmPred'):
                    f_name = feature_name.replace('_PgmPred', '')
                    f_name = f'{f_name}_{TAU}'
                    lstm_test_x[sample_idx, :, feature_idx] = inf_pred[f_name].iloc[j]
                    continue
                for seq_idx in range(TAU):
                    f_name = f'{feature_name}_{seq_idx}'
                    lstm_test_x[sample_idx, seq_idx, feature_idx] = group[f_name].iloc[j]

            for feature_idx, feature_name in enumerate(device_lstm_inout_feature_names[d_name]['out']):
                f_name = f'{feature_name}_{TAU}'
                lstm_test_y_pgm[sample_idx, feature_idx] = inf_pred[f_name].iloc[j]
                lstm_test_y_actual[sample_idx, feature_idx] = group[f_name].iloc[j]

            label = group['Label'].iloc[j]
            lstm_test_labels[sample_idx] = 1 if label == 'Abnormal' else 0
            onoff = group[f'{d_name}_0'].iloc[j]
            lstm_test_on_off[sample_idx] = onoff

            sample_idx += 1

        print(f'Inferencing {d_name} done.')

    assert sample_idx == len(dataset)
    np.save('data-dc/lstm_test_x.npy', lstm_test_x)  # (batch_size, seq_len, n_in_features)
    np.save('data-dc/lstm_test_y_pgm.npy', lstm_test_y_pgm)  # (batch_size, n_out_features)
    np.save('data-dc/lstm_test_y_actual.npy', lstm_test_y_actual)  # (batch_size, n_out_features)
    np.save('data-dc/lstm_test_labels.npy', lstm_test_labels)  # (batch_size,)
    np.save('data-dc/lstm_test_on_off.npy', lstm_test_on_off)  # (batch_size,)
    print('Saving done.')


if __name__ == '__main__':
    print('========Preprocessing test data...========')
    preprocess_test_data()
    print('========Preprocessing train data...========')
    preprocess_train_data()
    print('========Done.========')
