import sys
from collections import Counter
import matplotlib.pyplot as plt
import json
import bnlearn as bn
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import matplotlib
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from device_utils import state_extremas, device_id_to_name, device_name_to_id, get_device_type
from lstm_utils import load_device_lstm_inout_feature_names
from ml_utils import BinaryClassificationPerformance


device_lstm_inout_feature_names = load_device_lstm_inout_feature_names()
device_id_to_diff_normalization_factor = defaultdict(float)
for device_id in device_id_to_name.keys():
    out_features = device_lstm_inout_feature_names[device_id_to_name[device_id]]['out']
    for of in out_features:
        _, feature_type = of.split('_')
        if feature_type in state_extremas:
            device_id_to_diff_normalization_factor[device_id] += state_extremas[feature_type][1]
        else:
            print(f'feature_type: {feature_type} not found in state_extremas')
            exit(1)

matplotlib.use('Agg')  # 使用Agg后端生成图片

TAU = int(os.environ.get('TAU', 2))
print(f'TAU: {TAU}')

CALC_DIFF_METHOD = int(os.environ.get('CALC_DIFF_METHOD', 2))
print(f'CALC_DIFF_METHOD: {CALC_DIFF_METHOD}')

METHOD = os.environ.get('METHOD', 'a')
print(f'METHOD: {METHOD}')

devnull = open(os.devnull, 'w')

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


def build_graph(edges):
    graph = defaultdict(list)
    for edge in edges:
        graph[edge[0]].append(edge[1])
    return graph


def calc_input_columns(dataset: pd.DataFrame, DAG):
    input_columns = list(dataset.columns)
    dag_nodes = list(DAG['model'].nodes())

    input_column_excluded = set(input_columns) - set(dag_nodes)
    print(f'input_column_excluded: {input_column_excluded}')
    input_columns = list(set(input_columns) & set(dag_nodes))

    return input_columns


def get_influenced_nodes(graph: defaultdict, from_nodes: list[str]) -> list[str]:
    ret = set()

    def dfs(start_node: str, visited: set[str]):
        for child in graph[start_node]:
            if child not in visited:
                visited.add(child)
                ret.add(child)
                dfs(child, visited)

    visited = set()
    for node in from_nodes:
        visited.add(node)
        dfs(node, visited)

    froms = set(from_nodes)
    return list(ret - froms)


def get_other_influencing_nodes(group: pd.DataFrame, from_nodes: list[str], influenced_nodes: list[str]) -> list[str]:
    ground_truth_rules = {
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
    ground_truth_rules_rev = defaultdict(list)
    for k, v in ground_truth_rules.items():
        for vv in v:
            ground_truth_rules_rev[vv].append(k)

    other_influencing_nodes = []
    for inf_node in influenced_nodes:
        inf_room, inf_state, inf_seq = inf_node.split('_')
        for device_type in ground_truth_rules_rev[inf_state]:
            for col in group.columns:
                if device_type in col and inf_room in col and int(col.split('_')[-1]) < TAU:
                    other_influencing_nodes.append(col)

        if inf_room != 'Context':
            context_states = [f'Context_{inf_state}_{i}' for i in range(TAU)]
            other_influencing_nodes.extend(context_states)

    other_influencing_nodes = list(set(other_influencing_nodes) - set(from_nodes) - set(influenced_nodes))
    return other_influencing_nodes


def cleanise_dataset(dataset: pd.DataFrame, DAG, input_columns: list[str]) -> pd.DataFrame:
    # 获取模型中的所有节点
    nodes = DAG['model'].nodes()
    possible_values = {}

    # 遍历每个节点，获取其可能的状态值
    for node in nodes:
        cpd = DAG['model'].get_cpds(node)
        state_names = cpd.state_names[node]
        possible_values[node] = np.array(state_names)
        print(f'{node}: {possible_values[node]}')

    # 遍历每个输入列，将不在可能值列表中的值替换为最接近的可能值
    for ic in input_columns:
        dataset[ic] = dataset[ic].apply(lambda x: x if x in possible_values[ic] else possible_values[ic]
                                        [np.abs(possible_values[ic] - x).argmin()])

    return dataset


def extend_influenced_nodes(influenced_nodes: list[str]) -> list[str]:
    new_influenced_nodes = set()
    for ifn in influenced_nodes:
        room, property, seq = ifn.split('_')
        seq = int(seq)
        for i in range(seq + 1):
            new_influenced_nodes.add(f'{room}_{property}_{i}')
    return list(sorted(new_influenced_nodes))


def calc_diff_method_one(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes):
    this_input_columns = list(set(input_columns) - set(influenced_nodes))
    cases = dict()
    cases['from_nodes'] = group[from_nodes]
    cases['other_influencing_nodes'] = group[other_influencing_nodes]
    cases['Timestamp_0'] = group['Timestamp_0']
    cases[f'Timestamp_{TAU}'] = group[f'Timestamp_{TAU}']
    inf_pred_all = pd.DataFrame()
    inf_actual_all = pd.DataFrame()
    for inf_node in influenced_nodes:
        sys.stdout = devnull
        sys.stderr = devnull
        inf_pred = bn.predict(DAG, df=group[this_input_columns], variables=[inf_node])
        inf_pred = inf_pred[inf_node]
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        inf_actual = group[inf_node]
        inf_pred_all = pd.concat([inf_pred_all, inf_pred], axis=1)
        inf_actual_all = pd.concat([inf_actual_all, inf_actual], axis=1)

    cases['influenced_nodes_actual'] = inf_actual_all
    cases['influenced_nodes_pred'] = inf_pred_all
    influenced_nodes_last = [n for n in influenced_nodes if n.endswith(f'_{TAU}')]
    diff = np.abs(inf_pred_all[influenced_nodes_last].to_numpy() - inf_actual_all[influenced_nodes_last].to_numpy()).mean(axis=1)
    cases['diff'] = diff

    ret_cases = []
    n_lines = group.shape[0]
    for i in range(n_lines):
        this_case = {
            'from_nodes': cases['from_nodes'].iloc[i].to_dict(),
            'other_influencing_nodes': cases['other_influencing_nodes'].iloc[i].to_dict(),
            'influenced_nodes_actual': cases['influenced_nodes_actual'].iloc[i].to_dict(),
            'influenced_nodes_pred': cases['influenced_nodes_pred'].iloc[i].to_dict(),
            'diff': cases['diff'][i],
            'label': group['Label'].iloc[i],
            'on': group[from_nodes[0]].iloc[i] == 1,
            'Timestamp_0': cases['Timestamp_0'].iloc[i],
            f'Timestamp_{TAU}': cases[f'Timestamp_{TAU}'].iloc[i],
        }
        ret_cases.append(this_case)
    return ret_cases


def calc_diff_method_two(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    inf_pred = bn.predict(DAG, df=group[input_columns], variables=influenced_nodes)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    inf_actual = group[influenced_nodes]
    inf_pred = inf_pred[influenced_nodes]

    influenced_nodes_extended = extend_influenced_nodes(influenced_nodes)
    inf_extended = group[influenced_nodes_extended]

    influenced_nodes_last = [n for n in influenced_nodes if n.endswith(f'_{TAU}')]

    diff = np.abs(inf_pred[influenced_nodes_last].to_numpy() - inf_actual[influenced_nodes_last].to_numpy()).mean(axis=1)

    ret_cases = []
    n_lines = group.shape[0]
    for i in range(n_lines):
        this_case = {
            'from_nodes': group[from_nodes].iloc[i].to_dict(),
            'other_influencing_nodes': group[other_influencing_nodes].iloc[i].to_dict(),
            'influenced_nodes_actual': inf_actual.iloc[i].to_dict(),
            'influenced_nodes_extended': inf_extended.iloc[i].to_dict(),
            'influenced_nodes_pred': inf_pred.iloc[i].to_dict(),
            'diff': diff[i],
            'label': group['Label'].iloc[i],
            'on': group[from_nodes[0]].iloc[i] == 1,
            'Timestamp_0': group['Timestamp_0'].iloc[i],
            f'Timestamp_{TAU}': group[f'Timestamp_{TAU}'].iloc[i],
        }
        ret_cases.append(this_case)

    return ret_cases


def calc_diff_method_three(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    from_nodes_pred = bn.predict(DAG, df=group[input_columns], variables=from_nodes)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    from_nodes_actual = group[from_nodes]
    from_nodes_pred = from_nodes_pred[from_nodes]
    diff = np.abs(from_nodes_pred.to_numpy() - from_nodes_actual.to_numpy()).mean(axis=1)

    ret_cases = []
    n_lines = group.shape[0]
    for i in range(n_lines):
        this_case = {
            'other_influencing_nodes': group[other_influencing_nodes].iloc[i].to_dict(),
            'from_nodes_actual': from_nodes_actual.iloc[i].to_dict(),
            'from_nodes_pred': from_nodes_pred.iloc[i].to_dict(),
            'diff': diff[i],
            'label': group['Label'].iloc[i],
            'on': group[from_nodes[0]].iloc[i] == 1,
            'influenced_nodes_actual': group[influenced_nodes].iloc[i].to_dict(),
            'Timestamp_0': group['Timestamp_0'].iloc[i],
            f'Timestamp_{TAU}': group[f'Timestamp_{TAU}'].iloc[i],
        }
        ret_cases.append(this_case)
    return ret_cases


def calc_diff_method_four(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes):
    # group input columns by time step
    columns_grouped = [[] for _ in range(TAU + 1)]
    for ic in input_columns:
        _, _, seq = ic.split('_')
        columns_grouped[int(seq)].append(ic)

    # if there is influenced nodes in the timestep _0, we need to predict them first
    zero_influenced_nodes = [n for n in influenced_nodes if n.endswith('_0')]
    if zero_influenced_nodes:
        outt = group[columns_grouped[0]]
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        pred = bn.predict(DAG, df=outt, variables=zero_influenced_nodes)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        for node in zero_influenced_nodes:
            outt[node] = pred[node]
    else:
        outt = group[columns_grouped[0]]

    # predict _1 by _0, _2 by _1 and _0, and so on
    for i in range(1, TAU + 1):
        pred_ts = pd.DataFrame()
        for oc in columns_grouped[i]:
            if is_sensor(oc):
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                pred = bn.predict(DAG, df=outt, variables=[oc])
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                pred_ts[oc] = pred[oc]
            elif is_actuator(oc):
                pred_ts[oc] = group[oc]
        outt = pd.concat([outt, pred_ts], axis=1)

    diff = np.abs(outt[influenced_nodes].to_numpy() - group[influenced_nodes].to_numpy()).mean(axis=1)
    ret_cases = []
    n_lines = group.shape[0]
    for i in range(n_lines):
        this_case = {
            'from_nodes': group[from_nodes].iloc[i].to_dict(),
            'other_influencing_nodes': group[other_influencing_nodes].iloc[i].to_dict(),
            'influenced_nodes_actual': group[influenced_nodes].iloc[i].to_dict(),
            'influenced_nodes_pred': outt[influenced_nodes].iloc[i].to_dict(),
            'diff': diff[i],
            'label': group['Label'].iloc[i],
            'on': group[from_nodes[0]].iloc[i] == 1,
            'Timestamp_0': group['Timestamp_0'].iloc[i],
            f'Timestamp_{TAU}': group[f'Timestamp_{TAU}'].iloc[i],
        }
        ret_cases.append(this_case)
    return ret_cases


# 递归函数来处理嵌套的字典，并将 numpy 类型转换为原生类型
def convert_np_types(obj):
    if isinstance(obj, dict):
        return {key: convert_np_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # 这里将 numpy 类型转换为原生类型
    else:
        return obj


def evaluate():
    DAG = bn.load(f'data/DAG-{METHOD}.pkl')
    edges = list(DAG['model'].edges)
    graph = build_graph(edges)  # node_name -> [child_nodes]

    # 加载数据集
    dataset = pd.read_csv(f'data-dc/04_fault_detection_dataset_{TAU}.csv')
    dataset = dataset.reset_index(drop=True)

    # Predict based on the DAG
    input_columns = calc_input_columns(dataset, DAG)

    # cleanise dataset
    dataset = cleanise_dataset(dataset, DAG, input_columns)

    # diffs[<device_name>][<normal/abnormal>][<on/off>] = list of diffs
    diffs = dict()

    # wrong_examples[<device_name>][<normal/abnormal>][<on/off>] = list of wrong examples
    wrong_examples = dict()

    # correct_examples[<device_name>][<normal/abnormal>][<on/off>] = list of correct examples
    correct_examples = dict()

    # group dataset by device name
    dataset_grouped = dataset.groupby('Device')
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

    device_type_performance = defaultdict(BinaryClassificationPerformance)
    overall_performance = BinaryClassificationPerformance()

    for device_name, group in dataset_grouped:
        device_type = get_device_type(device_name)
        if device_type not in interested_device_types:
            continue
        device_id = device_name_to_id[device_name]
        diff_norm_factor = device_id_to_diff_normalization_factor[device_id]
        if device_name not in wrong_examples:
            wrong_examples[device_name] = dict()
            wrong_examples[device_name]['Normal'] = dict()
            wrong_examples[device_name]['Abnormal'] = dict()
            wrong_examples[device_name]['Normal']['On'] = []
            wrong_examples[device_name]['Normal']['Off'] = []
            wrong_examples[device_name]['Abnormal']['On'] = []
            wrong_examples[device_name]['Abnormal']['Off'] = []
            correct_examples[device_name] = dict()
            correct_examples[device_name]['Normal'] = dict()
            correct_examples[device_name]['Abnormal'] = dict()
            correct_examples[device_name]['Normal']['On'] = []
            correct_examples[device_name]['Normal']['Off'] = []
            correct_examples[device_name]['Abnormal']['On'] = []
            correct_examples[device_name]['Abnormal']['Off'] = []
        if device_name not in diffs:
            diffs[device_name] = dict()
            diffs[device_name]['Normal'] = dict()
            diffs[device_name]['Abnormal'] = dict()
            diffs[device_name]['Normal']['On'] = []
            diffs[device_name]['Normal']['Off'] = []
            diffs[device_name]['Abnormal']['On'] = []
            diffs[device_name]['Abnormal']['Off'] = []
            print(f'processing device: {device_name}')
        from_nodes = [f'{device_name}_{i}' for i in range(TAU + 1)]
        from_nodes = list(set(from_nodes) & set(input_columns))
        influenced_nodes = get_influenced_nodes(graph, from_nodes)
        if not influenced_nodes:
            print(f'uncheckable device: {device_name}, since it has no influenced nodes, maybe some problems in learing the DAG structure')
            continue
        other_influencing_nodes = get_other_influencing_nodes(group, from_nodes, influenced_nodes)
        other_influencing_nodes = list(sorted(set(other_influencing_nodes) & set(input_columns)))

        group = group.reset_index(drop=True)

        if CALC_DIFF_METHOD == 1:
            cases = calc_diff_method_one(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes)
        elif CALC_DIFF_METHOD == 2:
            if len(influenced_nodes) <= 10:
                cases = calc_diff_method_two(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes)
            else:
                cases = calc_diff_method_one(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes)
        elif CALC_DIFF_METHOD == 3:
            cases = calc_diff_method_three(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes)
        elif CALC_DIFF_METHOD == 4:
            cases = calc_diff_method_four(DAG, group, input_columns, from_nodes, influenced_nodes, other_influencing_nodes)
        for cas in cases:
            label = cas['label']
            diff = cas['diff']
            on_off = 'On' if cas['on'] else 'Off'
            diffs[device_name][label][on_off].append(diff)
            if diff > 0.0 and label == 'Normal':
                wrong_examples[device_name][label][on_off].append(cas)
            elif diff == 0.0 and label == 'Abnormal':
                wrong_examples[device_name][label][on_off].append(cas)
            elif diff == 0.0 and label == 'Normal':
                correct_examples[device_name][label][on_off].append(cas)
            elif diff > 0.0 and label == 'Abnormal':
                correct_examples[device_name][label][on_off].append(cas)

            y_true = 1 if label == 'Abnormal' else 0
            y_logit = diff / diff_norm_factor
            device_type_performance[device_type].add(y_true, y_logit)
            overall_performance.add(y_true, y_logit)

    wrong_examples = convert_np_types(wrong_examples)
    wrong_examples = merge_same_cases(wrong_examples)
    correct_examples = convert_np_types(correct_examples)
    correct_examples = merge_same_cases(correct_examples)
    output_figures(diffs)
    output_examples(wrong_examples, is_correct=False)
    output_examples(correct_examples, is_correct=True)
    
    out_table = pd.DataFrame(columns=['Device_Type', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
        
    best_threshold = overall_performance.get_best_threshold(method='youden_index')
    accuracy = overall_performance.get_accuracy(best_threshold)
    precision = overall_performance.get_precision(best_threshold)
    recall = overall_performance.get_recall(best_threshold)
    f1 = overall_performance.get_f1(best_threshold)
    auc = overall_performance.get_auc()
    print(f'Overall, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}')
    out_table.loc[len(out_table)] = ['Overall', accuracy, precision, recall, f1, auc]
    
    
    for device_type in device_type_performance:
        accuracy = device_type_performance[device_type].get_accuracy(best_threshold)
        precision = device_type_performance[device_type].get_precision(best_threshold)
        recall = device_type_performance[device_type].get_recall(best_threshold)
        f1 = device_type_performance[device_type].get_f1(best_threshold)
        auc = device_type_performance[device_type].get_auc()
        print(f'device_type: {device_type}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}')
        out_table.loc[len(out_table)] = [device_type, accuracy, precision, recall, f1, 0]
    
    # save the results
    out_table.to_csv('data-dc/evaluation_results_pgm.csv', index=False)


def output_examples(wrong_examples, is_correct=False):
    if is_correct:
        examples_base_dir = f'correct-examples-{METHOD}-{CALC_DIFF_METHOD}'
    else:
        examples_base_dir = f'wrong-examples-{METHOD}-{CALC_DIFF_METHOD}'
    os.makedirs(examples_base_dir, exist_ok=True)
    for device_name in wrong_examples:
        for label in wrong_examples[device_name]:
            with open(f'{examples_base_dir}/{device_name}-{label}.json5', 'w') as f:
                f.write(json.dumps(wrong_examples[device_name][label], indent=2, sort_keys=True))


def merge_same_cases(wrong_examples):
    for device_name in wrong_examples:
        for label in wrong_examples[device_name]:
            for on_off in wrong_examples[device_name][label]:
                cases = copy.deepcopy(wrong_examples[device_name][label][on_off])
                cases_no_timestamp = copy.deepcopy(wrong_examples[device_name][label][on_off])
                for cse in cases_no_timestamp:
                    cse.pop('Timestamp_0')
                    cse.pop(f'Timestamp_{TAU}')
                cases_no_timestamp = list({json.dumps(cse, sort_keys=True) for cse in cases_no_timestamp})
                cases_no_timestamp = [json.loads(cse) for cse in cases_no_timestamp]
                wrong_examples[device_name][label][on_off].clear()
                for cse in cases_no_timestamp:
                    freq = 0
                    flag = False
                    for ts_case in cases:
                        ts_0 = ts_case['Timestamp_0']
                        ts_tau = ts_case[f'Timestamp_{TAU}']
                        ts_case.pop('Timestamp_0')
                        ts_case.pop(f'Timestamp_{TAU}')
                        if ts_case == cse:  # hit!
                            freq += 1
                            if not flag:
                                ts_case['Timestamp_0'] = ts_0
                                ts_case[f'Timestamp_{TAU}'] = ts_tau
                                wrong_examples[device_name][label][on_off].append(ts_case)
                                flag = True
                            else:
                                ts_case['Timestamp_0'] = ts_0
                                ts_case[f'Timestamp_{TAU}'] = ts_tau
                        else:
                            ts_case['Timestamp_0'] = ts_0
                            ts_case[f'Timestamp_{TAU}'] = ts_tau
                    if flag:
                        wrong_examples[device_name][label][on_off][-1]['Frequency'] = freq
                wrong_examples[device_name][label][on_off] = sorted(
                    wrong_examples[device_name][label][on_off], key=lambda x: x['Timestamp_0'])
    return wrong_examples


def output_figures(diffs):
    figures_base_dir = f'figures-{METHOD}-{CALC_DIFF_METHOD}'
    os.makedirs(figures_base_dir, exist_ok=True)
    deep_green = (0.0, 0.5, 0.0)
    light_green = (0.0, 1.0, 0.0)
    deep_red = (0.5, 0.0, 0.0)
    light_red = (1.0, 0.0, 0.0)
    for device_name in diffs:
        normal_on_diffs = diffs[device_name]['Normal']['On']
        normal_off_diffs = diffs[device_name]['Normal']['Off']
        abnormal_on_diffs = diffs[device_name]['Abnormal']['On']
        abnormal_off_diffs = diffs[device_name]['Abnormal']['Off']
        normal_on_diffs_freq = Counter(normal_on_diffs)
        normal_off_diffs_freq = Counter(normal_off_diffs)
        abnormal_on_diffs_freq = Counter(abnormal_on_diffs)
        abnormal_off_diffs_freq = Counter(abnormal_off_diffs)

        y_true = [0] * len(normal_on_diffs) + [0] * len(normal_off_diffs) + [1] * len(abnormal_on_diffs) + [1] * len(abnormal_off_diffs)
        y_pred = normal_on_diffs + normal_off_diffs + abnormal_on_diffs + abnormal_off_diffs
        y_pred = [1 if y > 0.0 else 0 for y in y_pred]

        plt.figure()
        points = []
        for dif, cnt in normal_on_diffs_freq.items():
            points.append(dif)
            plt.text(dif, -0.05, f'{cnt}', color=deep_green, fontdict={'family': 'serif', 'weight': 'normal', 'size': 8})
        plt.plot(points, [0] * len(points), 'ro', color=deep_green, label='Normal On')
        points = []
        for dif, cnt in normal_off_diffs_freq.items():
            points.append(dif)
            plt.text(dif, 0.20, f'{cnt}', color=light_green, fontdict={'family': 'serif', 'weight': 'normal', 'size': 8})
        plt.plot(points, [0.25] * len(points), 'ro', color=light_green, label='Normal Off')
        points = []
        for dif, cnt in abnormal_on_diffs_freq.items():
            points.append(dif)
            plt.text(dif, 0.70, f'{cnt}', color=deep_red, fontdict={'family': 'serif', 'weight': 'normal', 'size': 8})
        plt.plot(points, [0.75] * len(points), 'ro', color=deep_red, label='Abnormal On')
        points = []
        for dif, cnt in abnormal_off_diffs_freq.items():
            points.append(dif)
            plt.text(dif, 0.95, f'{cnt}', color=light_red, fontdict={'family': 'serif', 'weight': 'normal', 'size': 8})

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        plt.text(0.0, 0.5, f'Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}', fontdict={
                 'family': 'serif', 'weight': 'normal', 'size': 8})
        plt.plot(points, [1] * len(points), 'ro', color=light_red, label='Abnormal Off')
        plt.xlabel('Value')
        plt.ylabel('Group')
        plt.legend()
        plt.title(f'Device: {device_name}')
        plt.savefig(f'{figures_base_dir}/{device_name}.png')
        plt.close()
        print(f'device: {device_name}, normal_on_diffs_freq: {dict(normal_on_diffs_freq)}, normal_off_diffs_freq: {dict(normal_off_diffs_freq)}, abnormal_on_diffs_freq: {dict(abnormal_on_diffs_freq)}, abnormal_off_diffs_freq: {dict(abnormal_off_diffs_freq)}')


def print_influenced_nodes():
    DAG = bn.load(f'data/DAG-{METHOD}.pkl')
    edges = list(DAG['model'].edges)
    graph = build_graph(edges)  # node_name -> [child_nodes]

    # 加载数据集
    dataset = pd.read_excel(f'data/03_time_series_windows_test_{TAU}.xlsx')
    drop_columns = [f'Timestamp_{i}' for i in range(TAU + 1)]
    dataset.drop(columns=drop_columns, inplace=True)

    # Predict based on the DAG
    input_columns = calc_input_columns(dataset, DAG)

    # cleanise dataset
    dataset = cleanise_dataset(dataset, DAG, input_columns)
    uncheckable_devices = set()

    # all devices
    all_devices = set()
    for col in dataset.columns:
        room, device, seq = col.split('_')
        if any([at in device for at in actuator_types]):
            all_devices.add(f'{room}_{device}')
    diffs = defaultdict(list)  # diffs[<device_name>] = list of diffs

    ret = {}

    for dev in all_devices:
        from_nodes = [f'{dev}_{i}' for i in range(TAU + 1)]
        from_nodes = list(set(from_nodes) & set(input_columns))
        influenced_nodes = get_influenced_nodes(graph, from_nodes)
        ret[dev] = influenced_nodes

    json.dump(ret, open('data-dc/influenced_nodes.json', 'w'))


if __name__ == '__main__':
    # learn_threshold()
    # print_influenced_nodes()
    evaluate()
