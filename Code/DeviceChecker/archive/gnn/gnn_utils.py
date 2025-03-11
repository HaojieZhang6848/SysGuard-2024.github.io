from typing import Dict, List, Tuple    # noqa
import queue
import numpy as np
from collections import defaultdict

room_name_to_id = {
    'Balcony': 1,
    'Bathroom': 2,
    'BedroomOne': 3,
    'BedroomTwo': 4,
    'Cloakroom': 5,
    'Context': 6,
    'Kitchen': 7,
    'LivingRoom': 8,
}

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


def is_actuator(name: str) -> bool:
    for at in actuator_types:
        if at in name:
            return True
    return False


state_name_to_id = defaultdict(lambda: 0)
state_name_to_id.update({
    'AirQuality': 1,
    'Brightness': 2,
    'COLevel': 3,
    'EnergyConsumption': 4,
    'HumanCount': 5,
    'HumanState': 6,
    'Humidity': 7,
    'Noise': 8,
    'Temperature': 9,
    'UltravioletLevel': 10,
    'Weather': 11,
})

device_name_to_id = defaultdict(lambda: 0)
device_name_to_id.update({
    'AC': 1,
    'AirPurifier': 2,
    'BathHeater': 3,
    'CookerHood': 4,
    'Curtain': 5,
    'Door': 6,
    'GasStove': 7,
    'Heater': 8,
    'Humidifier': 9,
    'Light': 10,
    'MicrowaveOven': 11,
    'Printer': 12,
    'Projector': 13,
    'Speaker': 14,
    'TowelDryer': 15,
    'TV': 16,
    'WashingMachine': 17,
    'WaterDispenser': 18,
    'WaterHeater': 19,
    'Window': 20,
})


class NodeAttr:
    pred_val: int
    actual_val: int
    room_name: int
    device_name: int
    state_name: int
    seq_num: int

    def __init__(self, pred_val: int, actual_val: int, room_name_str: str, device_name_str: str, state_name_str: str, seq_num: int):
        self.pred_val = pred_val
        self.actual_val = actual_val
        self.room_name = room_name_to_id[room_name_str]
        self.device_name = device_name_to_id[device_name_str]
        self.state_name = state_name_to_id[state_name_str]
        self.seq_num = seq_num


class DCSubGraph:
    device_name: str
    from_nodes: List[str]
    influenced_nodes: List[str]
    nodes: Dict[str, NodeAttr]  # node_name -> NodeAttr
    edges: List[Tuple[str, str]]  # (src_node_name, dst_node_name)

    def __init__(self):
        pass

    @staticmethod
    def build(device_name: str, DAG):
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

        subgraph = DCSubGraph()
        subgraph.device_name = device_name
        subgraph.from_nodes = list(sorted(from_nodes))
        subgraph.influenced_nodes = list(sorted(influenced_nodes))
        subgraph.nodes = {}
        for node in subgraph_nodes:
            room_name, device_or_state_str, seq_num = node.split('_')
            seq_num = int(seq_num)
            if is_actuator(device_or_state_str):
                device_name = device_or_state_str[:-3]
                state_name = 'Default'
            else:
                device_name = 'Default'
                state_name = device_or_state_str
            subgraph.nodes[node] = NodeAttr(0, 0, room_name, device_name, state_name, seq_num)

        subgraph.edges = list(sorted(subgraph_edges))
        return subgraph

    def to_gnn_input(self):
        # gnn input: x, edge_index
        # x: [num_nodes, num_node_features]
        # edge_index: [2, num_edges]
        num_node_features = 6
        node_2_id = {node: i for i, node in enumerate(sorted(self.nodes.keys()))}  # node_id -> node_index
        n_nodes = len(node_2_id)
        n_edges = len(self.edges)
        x = np.zeros((n_nodes, num_node_features), dtype=np.float32)
        edge_index = np.zeros((2, n_edges), dtype=np.int64)
        for node in self.nodes:
            node_index = node_2_id[node]
            x[node_index, 0] = self.nodes[node].pred_val
            x[node_index, 1] = self.nodes[node].actual_val
            x[node_index, 2] = self.nodes[node].room_name
            x[node_index, 3] = self.nodes[node].device_name
            x[node_index, 4] = self.nodes[node].state_name
            x[node_index, 5] = self.nodes[node].seq_num
        for i, edge in enumerate(self.edges):
            edge_index[0, i] = node_2_id[edge[0]]
            edge_index[1, i] = node_2_id[edge[1]]
        return x, edge_index
