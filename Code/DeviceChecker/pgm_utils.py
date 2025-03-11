import pandas as pd
import numpy as np
import queue
import sys
import os
import bnlearn as bn


def calc_valid_columns(dataset: pd.DataFrame, DAG):
    input_columns = list(dataset.columns)
    dag_nodes = list(DAG['model'].nodes())
    input_columns = list(set(input_columns) & set(dag_nodes))
    return input_columns


def cleanise_dataset(dataset: pd.DataFrame, DAG, valid_columns: list[str]) -> pd.DataFrame:
    # 获取模型中的所有节点
    nodes = DAG['model'].nodes()
    possible_values = {}

    # 遍历每个节点，获取其可能的状态值
    for node in nodes:
        cpd = DAG['model'].get_cpds(node)
        state_names = cpd.state_names[node]
        possible_values[node] = np.array(state_names)

    # 遍历每个输入列，将不在可能值列表中的值替换为最接近的可能值
    for ic in valid_columns:
        dataset[ic] = dataset[ic].apply(lambda x: x if x in possible_values[ic] else possible_values[ic]
                                        [np.abs(possible_values[ic] - x).argmin()])
    return dataset


def calc_influenced_nodes(device_name: str, DAG) -> list[str]:
    dag_nodes = DAG['model'].nodes
    dag_edges = DAG['model'].edges

    from_nodes = [node for node in dag_nodes if node.startswith(device_name)]
    subgraph_nodes = set(from_nodes)
    subgraph_edges = set()

    # breadth-first search
    for fn in from_nodes:
        visited = set()
        q = queue.Queue()
        q.put(fn)
        visited.add(fn)
        while not q.empty():
            node = q.get()
            for edge in dag_edges:
                if edge[0] == node:
                    subgraph_edges.add(edge)
                    if edge[1] not in visited:
                        q.put(edge[1])
                        visited.add(edge[1])
                        subgraph_nodes.add(edge[1])

    influenced_nodes = set(set(subgraph_nodes) - set(from_nodes))
    influenced_nodes = list(sorted(influenced_nodes))
    return influenced_nodes


def calc_influenced_nodes_value(DAG, dataset, valid_columns, influenced_nodes):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    inf_pred = bn.predict(DAG, df=dataset[valid_columns], variables=influenced_nodes)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    inf_pred = inf_pred[influenced_nodes]
    return inf_pred
