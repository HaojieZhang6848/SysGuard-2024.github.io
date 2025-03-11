import bnlearn as bn
import pandas as pd
import os
from gnn_utils import DCSubGraph
import json
import sys
import numpy as np
import pickle
import random

random.seed(42)
np.random.seed(42)

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

TAU = os.getenv('TAU', 2)
DAG_METHOD = os.getenv('DAG_METHOD', 'predefined-edges')

with open('data/device_names.json', 'r') as f:
    device_names = json.load(f)


def cleanise_dataset(dataset: pd.DataFrame, DAG, input_columns: list[str]) -> pd.DataFrame:
    # 获取模型中的所有节点
    nodes = DAG['model'].nodes()
    possible_values = {}

    # 遍历每个节点，获取其可能的状态值
    for node in nodes:
        cpd = DAG['model'].get_cpds(node)
        state_names = cpd.state_names[node]
        possible_values[node] = np.array(state_names)

    # 遍历每个输入列，将不在可能值列表中的值替换为最接近的可能值
    for ic in input_columns:
        dataset[ic] = dataset[ic].apply(lambda x: x if x in possible_values[ic] else possible_values[ic]
                                        [np.abs(possible_values[ic] - x).argmin()])
    return dataset


def calc_input_columns(dataset: pd.DataFrame, DAG):
    input_columns = list(dataset.columns)
    dag_nodes = list(DAG['model'].nodes())
    input_columns = list(set(input_columns) & set(dag_nodes))
    return input_columns


def calc_influenced_nodes_pred(DAG, group, input_columns, influenced_nodes):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    inf_pred = bn.predict(DAG, df=group[input_columns], variables=influenced_nodes)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    inf_pred = inf_pred[influenced_nodes]

    return inf_pred


def main():
    DAG = bn.load(f'data/DAG-{DAG_METHOD}.pkl')
    time_series_windows_train1 = pd.read_excel(f'data/03_time_series_windows_train1_{TAU}.xlsx')
    time_series_windows_train2 = pd.read_excel(f'data/03_time_series_windows_train2_{TAU}.xlsx')
    time_series_windows_train = pd.concat([time_series_windows_train1, time_series_windows_train2])
    input_columns = calc_input_columns(time_series_windows_train, DAG)
    time_series_windows_train = cleanise_dataset(time_series_windows_train, DAG, input_columns)

    fault_detection_dataset = pd.read_csv(f'data-dc/04_fault_detection_dataset_{TAU}.csv')
    fault_detection_dataset = cleanise_dataset(fault_detection_dataset, DAG, input_columns)

    training_preprocessed = []
    testing_preprocessed = []

    for d_name in device_names:
        _, name = d_name.split('_')
        d_type = name[:-3]
        if d_type not in interested_device_types:
            print(f'Skipping {d_name}, since we are not interested in this device type')
            continue
        subgraph = DCSubGraph.build(d_name, DAG)
        n_nodes = len(subgraph.nodes)
        n_edges = len(subgraph.edges)
        if n_nodes == 0 or n_edges == 0:
            print(f'Skipping {d_name}, since subgraph has no nodes or edges')
            continue

        print(f'Preprocessing for {d_name}')
        from_nodes = subgraph.from_nodes
        influenced_nodes = subgraph.influenced_nodes

        # building testing preprocessed
        test_samples = fault_detection_dataset[fault_detection_dataset['Device'] == d_name]
        # test_samples = filter_by_all_same_from_nodes(test_samples, from_nodes)
        n_test_normal = test_samples[test_samples['Label'] == 'Normal'].shape[0]
        n_test_abnormal = test_samples[test_samples['Label'] == 'Abnormal'].shape[0]

        inf_pred = calc_influenced_nodes_pred(DAG, test_samples, input_columns, influenced_nodes)
        inf_actual = test_samples[influenced_nodes]
        for i in range(len(test_samples)):
            for node in subgraph.nodes:
                if node in influenced_nodes:
                    subgraph.nodes[node].actual_val = inf_actual[node].iloc[i]
                    subgraph.nodes[node].pred_val = inf_pred[node].iloc[i]
                else:
                    subgraph.nodes[node].actual_val = test_samples[node].iloc[i]
                    subgraph.nodes[node].pred_val = test_samples[node].iloc[i]
            x, edge_index = subgraph.to_gnn_input()
            testing_preprocessed.append((x, edge_index, 0 if test_samples['Label'].iloc[i] == 'Normal' else 1, d_name))

        # build training preprocessed
        tsw_filtered = time_series_windows_train
        # tsw_filtered = filter_by_all_same_from_nodes(time_series_windows_train, from_nodes)
        n = (n_test_normal + n_test_abnormal) // 2 * 10
        n_train_normal = n
        n_train_abnormal = n
        train_samples = tsw_filtered.sample(n=n)
        inf_pred = calc_influenced_nodes_pred(DAG, train_samples, input_columns, influenced_nodes)
        inf_actual = train_samples[influenced_nodes]
        for i in range(n):
            # normal device execution
            for node in subgraph.nodes:
                if node in influenced_nodes:
                    subgraph.nodes[node].actual_val = inf_actual[node].iloc[i]
                    subgraph.nodes[node].pred_val = inf_pred[node].iloc[i]
                else:
                    subgraph.nodes[node].actual_val = train_samples[node].iloc[i]
                    subgraph.nodes[node].pred_val = train_samples[node].iloc[i]
            x, edge_index = subgraph.to_gnn_input()
            training_preprocessed.append((x, edge_index, 0))  # 0 for normal device execution

            # since we are building an one-class classification model, we only have normal device execution data
            # we inject some noise to the actual values to simulate abnormal device execution in the training dataset
            for node in subgraph.nodes:
                if node in influenced_nodes:
                    _, state, _ = node.split('_')
                    node_actual = inf_actual.iloc[i][node]
                    node_pred = inf_pred.iloc[i][node]
                    state_max = state_extremas[state][1]
                    state_min = state_extremas[state][0]
                    dist_to_max = abs(state_max - node_actual)
                    dist_to_min = abs(node_actual - state_min)
                    node_actual_noised = state_max if dist_to_max > dist_to_min else state_min
                    subgraph.nodes[node].actual_val = node_actual_noised
                    subgraph.nodes[node].pred_val = node_pred
                else:
                    subgraph.nodes[node].actual_val = train_samples[node].iloc[i]
                    subgraph.nodes[node].pred_val = train_samples[node].iloc[i]

            x, edge_index = subgraph.to_gnn_input()
            training_preprocessed.append((x, edge_index, 1))  # 1 for abnormal device execution

        print(
            f'Finished building dataset for {d_name}, {n_train_normal} normal samples and {n_train_abnormal} abnormal samples for training, {n_test_normal} normal samples and {n_test_abnormal} abnormal samples for testing')

    random.shuffle(training_preprocessed)
    random.shuffle(testing_preprocessed)
    pickle.dump(training_preprocessed, open(f'data-dc/05_training_preprocessed_gnn_{TAU}.pkl', 'wb'))
    pickle.dump(testing_preprocessed, open(f'data-dc/05_testing_preprocessed_gnn_{TAU}.pkl', 'wb'))


def filter_by_all_same_from_nodes(time_series_windows_train, from_nodes):
    tsw_filtered = time_series_windows_train[time_series_windows_train.apply(
        lambda row: all(row[from_nodes[0]] == row[fn] for fn in from_nodes), axis=1)]

    return tsw_filtered


if __name__ == '__main__':
    main()
