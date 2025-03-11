from collections import defaultdict
import json
import queue
import os

TAU = int(os.getenv('TAU', 2))

def calc_save_device_lstm_inout_feature_names(DAG, device_names):
    # device_inout_feature_names[<device_name>]['in'] = [<input_feature_names>], device_inout_feature_names[<device_name>]['out'] = [<output_feature_names>]
    device_lstm_inout_feature_names = defaultdict(lambda: {'in': [], 'out': []})
    for d_name in device_names:
        in_feature_names, out_feature_names = calc_lstm_inout_feature_names(DAG, d_name)
        device_lstm_inout_feature_names[d_name]['in'] = in_feature_names
        device_lstm_inout_feature_names[d_name]['out'] = out_feature_names
    with open('data-dc/device_lstm_inout_feature_names.json', 'w') as f:
        json.dump(device_lstm_inout_feature_names, f, sort_keys=True, indent=2)
    return device_lstm_inout_feature_names

def load_device_lstm_inout_feature_names():
    with open('data-dc/device_lstm_inout_feature_names.json', 'r') as f:
        device_lstm_inout_feature_names = json.load(f)
    return device_lstm_inout_feature_names

def calc_lstm_inout_feature_names(DAG, device_name):
    in_feature_names = set()
    out_feature_names = set()
    
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

    # rev breadth-first search
    influenced_nodes = set(set(subgraph_nodes) - set(from_nodes))
    for ifn in influenced_nodes:
        visited = set()
        q = queue.Queue()
        q.put(ifn)
        visited.add(ifn)
        while not q.empty():
            node = q.get()
            for edge in dag_edges:
                if edge[1] == node:
                    subgraph_edges.add(edge)
                    if edge[0] not in visited:
                        q.put(edge[0])
                        visited.add(edge[0])
                        subgraph_nodes.add(edge[0])
                        
    for node in subgraph_nodes:
        room_name, device_or_state_str, seq_num = node.split('_')
        seq_num = int(seq_num)
        if seq_num < TAU:
            in_feature_names.add(f'{room_name}_{device_or_state_str}')
        else:
            out_feature_names.add(f'{room_name}_{device_or_state_str}')
            
    in_feature_names = list(sorted(in_feature_names))
    out_feature_names = list(sorted(out_feature_names))
    
    influenced_nodes = list(filter(lambda x: x.endswith(f'_{TAU}'), influenced_nodes))
    for ifn in influenced_nodes:
        room_name, state_attr, _ = ifn.split('_')
        in_feature_names.insert(0, f'{room_name}_{state_attr}_PgmPred')
    in_feature_names.insert(0, 'device_id')
    
    return in_feature_names, out_feature_names