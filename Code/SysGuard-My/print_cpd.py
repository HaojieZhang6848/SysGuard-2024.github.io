import bnlearn as bn
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD

def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup

# 设置打印选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

model = bn.load('data/DAG-predefined-edges.pkl')
for node in model['model'].nodes():
    cpd = model['model'].get_cpds(node)
    print(f'cpd of {node}')
    print_full(cpd)