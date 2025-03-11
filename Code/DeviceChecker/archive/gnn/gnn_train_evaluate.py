from calendar import c
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import pickle
import os
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict
from dl_utils import ModelPerformance, EarlyStopping
import optuna

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TAU = os.getenv('TAU', 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(f'data-dc/05_training_preprocessed_gnn_{TAU}.pkl', 'rb') as f:
    training_dataset = pickle.load(f)
with open(f'data-dc/05_testing_preprocessed_gnn_{TAU}.pkl', 'rb') as f:
    testing_dataset = pickle.load(f)

bar = int(0.8 * len(testing_dataset))
validating_dataset = testing_dataset[bar:]
testing_dataset = testing_dataset[:bar]


class GNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GNN_Model, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, node_features, edge_index):
        x = self.conv1(node_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.conv2(x, edge_index)
        x = x.mean(dim=0)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.fc(x)
        return x


def train(model, optimizer, scheduler, criterion, training_dataset, validating_dataset, epoch, early_stopping):
    highest_auc = 0
    for epoch_id in range(1, epoch + 1):
        model.train()
        random.shuffle(training_dataset)
        total_loss = 0
        for x, edge_index, label in tqdm(training_dataset):
            x = torch.tensor(x, dtype=torch.float).to(device)
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
            label = torch.tensor(label).to(device)
            optimizer.zero_grad()
            output = model(x, edge_index)
            loss = criterion(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        total_loss /= len(training_dataset)
        print(f'Epoch {epoch_id} training loss: {loss.item()}')

        model.eval()
        total_loss = 0
        performance = ModelPerformance()
        with torch.no_grad():
            for x, edge_index, label, _ in tqdm(validating_dataset):
                x = torch.tensor(x, dtype=torch.float).to(device)
                edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
                label = torch.tensor(label).to(device)
                output = model(x, edge_index)
                loss = criterion(output, label)
                total_loss += loss.item()
                performance.add(label.item(), output.argmax().item(), torch.nn.functional.softmax(output, dim=0)[1].item())
        total_loss /= len(validating_dataset)

        accuracy = performance.get_accuracy()
        recall = performance.get_recall()
        precision = performance.get_precision()
        f1 = performance.get_f1()
        auc = performance.get_auc()
        highest_auc = max(highest_auc, auc)
        print(f'Epoch {epoch_id} validating loss: {total_loss}, accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}, AUC: {auc}')

        early_stopping(-auc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(total_loss)

    return highest_auc


def test(model, testing_dataset, criterion):
    model.eval()
    total_loss = 0

    device_performances = defaultdict(ModelPerformance)  # device_name -> ModelPerformance
    type_performances = defaultdict(ModelPerformance)  # device_type -> ModelPerformance
    overall_performance = ModelPerformance()

    with torch.no_grad():
        for x, edge_index, label, device_name in tqdm(testing_dataset):
            device_type = device_name.split('_')[1][:-3]
            x = torch.tensor(x, dtype=torch.float).to(device)
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
            label = torch.tensor(label).to(device)
            output = model(x, edge_index)
            loss = criterion(output, label)
            total_loss += loss.item()
            label_true = label.item()
            # label_pred = output.argmax().item()
            label_pred_prob = torch.nn.functional.softmax(output, dim=0)[1].item()
            label_pred = 1 if label_pred_prob > 0.1 else 0

            overall_performance.add(label_true, label_pred, label_pred_prob)
            device_performances[device_name].add(label_true, label_pred, label_pred_prob)
            type_performances[device_type].add(label_true, label_pred, label_pred_prob)

    print("==============Performance By Device Name==============")
    for d_name, perf in device_performances.items():
        accuracy = perf.get_accuracy()
        recall = perf.get_recall()
        precision = perf.get_precision()
        f1 = perf.get_f1()
        auc = perf.get_auc()
        print(f'{d_name}: accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}, AUC: {auc}')

    print("==============Performance By Device Type==============")
    for d_type, perf in type_performances.items():
        accuracy = perf.get_accuracy()
        recall = perf.get_recall()
        precision = perf.get_precision()
        f1 = perf.get_f1()
        auc = perf.get_auc()
        print(f'{d_type}: accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}, AUC: {auc}')

    print("==============Overall Performance==============")
    total_loss /= len(testing_dataset)
    accuracy = overall_performance.get_accuracy()
    recall = overall_performance.get_recall()
    precision = overall_performance.get_precision()
    f1 = overall_performance.get_f1()
    auc = overall_performance.get_auc()
    print(f'Overall: accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}, AUC: {auc}, loss: {total_loss}')
    print(f"best threshold: {overall_performance.get_best_threshold()}")


def main():
    input_dim = 6
    hidden_dim = 32
    output_dim = 2
    model = GNN_Model(input_dim, hidden_dim, output_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    epoch = 100
    earyly_stopping = EarlyStopping(patience=10, verbose=True)
    train(model, optimizer, scheduler, criterion, training_dataset, validating_dataset, epoch, earyly_stopping)

    # load
    model.load_state_dict(torch.load('gnn_checkpoint.pt'))
    test(model, testing_dataset, criterion)


def objective(trail):
    n_hidden_dim = trail.suggest_int('n_hidden_dim', 16, 128)
    learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trail.suggest_uniform('dropout_rate', 0.1, 0.9)
    trail_id = trail.number
    
    print(f'====================Trail {trail_id}, n_hidden_dim: {n_hidden_dim}, learning_rate: {learning_rate}, dropout_rate: {dropout_rate}====================')

    input_dim = 6
    output_dim = 2
    model = GNN_Model(input_dim, n_hidden_dim, output_dim, dropout_rate)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()
    epoch = 100
    early_stopping = EarlyStopping(patience=5, verbose=True)

    highest_auc = train(model, optimizer, scheduler, criterion, training_dataset, validating_dataset, epoch, early_stopping)
    return highest_auc


def hyperparameter_search():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print("Best hyperparameters: ", study.best_params)


if __name__ == '__main__':
    hyperparameter_search()
