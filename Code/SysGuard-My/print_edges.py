import bnlearn as bn

model = bn.load('data/DAG-predefined-edges.pkl')
edges = model['model'].edges()
edges = list(sorted(edges, key=lambda x: x[0]))
for edge in edges:
    print(edge)