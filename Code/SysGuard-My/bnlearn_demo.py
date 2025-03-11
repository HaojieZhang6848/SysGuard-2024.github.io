import bnlearn as bn
import pandas as pd

# 1. 生成示例数据
data = pd.DataFrame({
    'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Cold', 'Warm', 'Hot', 'Cold', 'Warm', 'Hot', 'Cold'],
    'Activity': ['Beach', 'Stay Home', 'Park', 'Beach', 'Stay Home', 'Park', 'Beach', 'Stay Home']
})

# 2. 结构学习（从数据中学习贝叶斯网络）
model = bn.structure_learning.fit(data, methodtype='hc', scoretype='bic')

# 3. 参数学习（计算条件概率分布）
model = bn.parameter_learning.fit(model, data)
bn.save(model, 'data/bnlearn-model.pkl')

# 4. 预测：已知 Weather='Sunny'，预测 Temperature 和 Activity
query_df = pd.DataFrame({'Activity': ['Beach']})
predicted_result = bn.inference.fit(model, evidence={'Activity': 'Beach'}, variables=['Weather', 'Temperature'])
predicted_result_2 = bn.predict(model, query_df, variables=['Weather', 'Temperature'])

# 打印预测结果
print(predicted_result)
print("----")
print(predicted_result_2)