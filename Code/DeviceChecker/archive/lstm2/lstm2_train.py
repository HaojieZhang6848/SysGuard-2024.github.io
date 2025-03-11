from calendar import c
import json
import random
import bnlearn as bn
import os

import pandas as pd
import torch
from zmq import device
from other_utils import get_interested_device_names, get_device_type
from lstm2_utils import load_device_lstm2_inout_feature_names
from pgm_utils import cleanise_dataset, calc_valid_columns, calc_influenced_nodes, calc_influenced_nodes_value
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import torch.nn as nn
import optuna
from dl_utils import Lstm2Dataset
from dl_utils import EarlyStopping, train

np.random.seed(42)
random.seed(42)

lstm2_train_x = np.load('data-dc/lstm2_train_x.npy')
lstm2_train_labels = np.load('data-dc/lstm2_train_labels.npy')
lstm2_test_x = np.load('data-dc/lstm2_test_x.npy')
lstm2_test_labels = np.load('data-dc/lstm2_test_labels.npy')

train_dataset = Lstm2Dataset(lstm2_train_x, lstm2_train_labels)
test_dataset = Lstm2Dataset(lstm2_test_x, lstm2_test_labels)

device_names = get_interested_device_names()
device_lstm2_in_feture_names = load_device_lstm2_inout_feature_names()

n_in_features = max([len(device_lstm2_in_feture_names[device_name]['in']) for device_name in device_names])  # 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TAU = int(os.getenv('TAU', 2))


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_lstm_layers=2, dropout_rate=0.5):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_lstm_layers, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.droupout(x)
        x = self.fc(x)
        return x


def objective(trail):
    n_hidden_features = trail.suggest_int('n_hidden_features', 32, 256)
    dropout_rate = trail.suggest_float('dropout_rate', 0.1, 0.5)
    num_lstm_layers = trail.suggest_int('num_lstm_layers', 1, 3)
    batch_size = trail.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    lr = trail.suggest_loguniform('lr', 1e-5, 1e-2)
    trail_id = trail.number

    print(f'========== Trail {trail_id}, n_hidden_features: {n_hidden_features}, dropout_rate: {dropout_rate}, num_lstm_layers: {num_lstm_layers}, batch_size: {batch_size}, lr: {lr} ==========')

    model = LSTM_Model(n_in_features, n_hidden_features, 2, num_lstm_layers, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
    early_stopping = EarlyStopping(patience=15, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = 100

    return train(model, optimizer, scheduler, criterion, train_loader, val_loader, epochs, early_stopping, device)


def hyperparam_search():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print('Best trial:')
    print(study.best_trial)
    print('Best params:')
    print(study.best_params)


if __name__ == '__main__':
    hyperparam_search()
