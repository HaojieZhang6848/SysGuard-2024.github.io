import glob
from sympy import hyper
import torch.nn as nn
import torch.optim
import numpy as np
import random
from torch.utils.data import DataLoader
import os
import optuna

from lstm_utils import load_device_lstm_inout_feature_names
from device_utils import get_interested_device_names
from ml_utils import EarlyStopping, LSTM_Dataset, train_time_series_forecast

device_lstm_inout_feature_names = load_device_lstm_inout_feature_names()
device_names = get_interested_device_names()
device_id_to_name = {i: d_name for i, d_name in enumerate(device_names)}
device_name_to_id = {d_name: i for i, d_name in enumerate(device_names)}

device_id_to_n_valid_out_features = {device_id: len(
    device_lstm_inout_feature_names[device_id_to_name[device_id]]['out']) for device_id in device_id_to_name.keys()}

device_id_to_mask_feature_ids = {device_id: [] for device_id in device_id_to_name.keys()}
for device_id in device_id_to_name.keys():
    device_name = device_id_to_name[device_id]
    in_feature_names = device_lstm_inout_feature_names[device_name]['in']
    out_feature_names = device_lstm_inout_feature_names[device_name]['out']
    mask_feature_ids = []
    for idx, if_name in enumerate(in_feature_names):
        for _, of_name in enumerate(out_feature_names):
            if if_name == of_name:
                mask_feature_ids.append(idx)
                break


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_in_features = max([len(device_lstm_inout_feature_names[device_name]['in']) for device_name in device_names])  # 9
n_out_features = max([len(device_lstm_inout_feature_names[device_name]['out']) for device_name in device_names])  # 2
n_hidden_features = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TAU = int(os.getenv('TAU', 2))


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_lstm_layers=2, dropout_rate=0.5):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        if self.training:
            device_ids = x[:, 0, 0]
            for dev_id in torch.unique(device_ids):
                dev_mask = device_ids == dev_id
                mask_feature_ids = device_id_to_mask_feature_ids[int(dev_id)]
                for mask_feature_id in mask_feature_ids:
                    mask = torch.bernoulli(torch.full_like(x[dev_mask, :, mask_feature_id], 1 - self.dropout_rate))
                    x[dev_mask, :, mask_feature_id] = x[dev_mask, :, mask_feature_id] * mask
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.droupout(x)
        x = self.fc(x)
        return x


lstm_x = np.load('data-dc/lstm_train_x.npy')
lstm_y = np.load('data-dc/lstm_train_y_diff.npy')

bar_1 = int(len(lstm_x) * 0.8)
x_train, x_val = lstm_x[:bar_1], lstm_x[bar_1:]
y_train, y_val = lstm_y[:bar_1], lstm_y[bar_1:]
train_dataset, val_dataset = LSTM_Dataset(x_train, y_train), LSTM_Dataset(x_val, y_val)


class DeviceCheckerMseLoss(nn.Module):
    def __init__(self):
        super(DeviceCheckerMseLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y, y_pred):
        device_ids = x[:, 0, 0]
        for device_id in torch.unique(device_ids):
            mask = device_ids == device_id
            out_feature_num = device_id_to_n_valid_out_features[int(device_id)]
            y[mask, out_feature_num:] = 0
            y_pred[mask, out_feature_num:] = 0
        return self.mse_loss(y, y_pred)


def train_stage():
    n_hidden_features = 106
    batch_size = 64
    dropout_rate = 0.5642639874269533
    learning_rate =  0.006485012226865641
    num_lstm_layers = 2

    model = LSTM_Model(n_in_features, n_hidden_features, n_out_features,
                       num_lstm_layers=num_lstm_layers, dropout_rate=dropout_rate).to(device)
    criterion = DeviceCheckerMseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=1e-6, save_path='data-dc/lstm_model_final2.pth')
    num_epochs = 200

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8)

    train_time_series_forecast(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs, early_stopping, device)

hyperparameter_search_instance = 0

def objective(trail):
    global hyperparameter_search_instance
    n_hidden_features = trail.suggest_int('n_hidden_features', 16, 128)
    batch_size = trail.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
    dropout_rate = trail.suggest_uniform('dropout_rate', 0.1, 0.6)
    learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_lstm_layers = trail.suggest_int('num_lstm_layers', 1, 3)

    trail_id = trail.number
    print(f'====================Trail {trail_id}, n_hidden_features: {n_hidden_features}, batch_size: {batch_size}, dropout_rate: {dropout_rate}, learning_rate: {learning_rate}, num_lstm_layers: {num_lstm_layers}====================')

    model = LSTM_Model(n_in_features, n_hidden_features, n_out_features,
                       num_lstm_layers=num_lstm_layers, dropout_rate=dropout_rate).to(device)
    criterion = DeviceCheckerMseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=1e-6, save_path=f'data-dc/lstm_model_{hyperparameter_search_instance}_{trail_id}.pth')
    num_epochs = 200

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8)
    highest_auc = train_time_series_forecast(model, optimizer, scheduler, criterion, train_loader,
                                             val_loader, num_epochs, early_stopping, device)
    return highest_auc


def hyperparameter_search():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print("Best hyperparameters: ", study.best_params)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == 'hyperparameter_search':
        instance = sys.argv[2]
        hyperparameter_search_instance = int(instance)
        print(f'Hyperparameter search {instance}...')
        hyperparameter_search()
    else:
        train_stage()
