import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 构造一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义目标函数
def objective(trial):
    # 超参数搜索空间
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)

    # 构造数据集
    X = torch.rand(1000, 10)
    y = torch.rand(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 构建模型
    model = SimpleModel(input_dim=10, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(5):  # 训练 5 个 epoch
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

    return loss.item()  # 目标函数返回最小损失值

# 使用 Optuna 进行超参数优化
study = optuna.create_study(direction='minimize')  # 最小化 loss
study.optimize(objective, n_trials=20)  # 进行 20 次搜索

# 输出最优超参数
print("最佳超参数:", study.best_params)
