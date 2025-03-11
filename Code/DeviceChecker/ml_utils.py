import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve


class LSTM_Dataset(Dataset):
    def __init__(self, lstm_x, lstm_y):
        self.lstm_x = lstm_x
        self.lstm_y = lstm_y

    def __len__(self):
        return len(self.lstm_x)

    def __getitem__(self, idx):
        return self.lstm_x[idx], self.lstm_y[idx]


class DeviceCheckerDataset(Dataset):
    def __init__(self, x, y_pgm, y_actual, labels, on_off):
        self.x = x
        self.y_pgm = y_pgm
        self.y_actual = y_actual
        self.labels = labels
        self.on_off = on_off

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_pgm[idx], self.y_actual[idx], self.labels[idx], self.on_off[idx]


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_path='data-dc/lstm_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


def train_time_series_forecast(model, optimizer, scheduler, criterion, train_loader, val_loader, epoch, early_stopping, device):
    from lstm_evaluate import eval_auc
    min_val_loss = np.Inf
    for epoch_id in range(1, epoch + 1):
        model.train()
        train_loss = 0
        for train_x, train_y in tqdm(train_loader):
            train_x, train_y = train_x.to(device).float(), train_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(train_x)
            loss = criterion(train_x, train_y, outputs)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        print(f'Epoch {epoch_id} training loss: {train_loss}')

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_x, val_y in tqdm(val_loader):
                val_x, val_y = val_x.to(device).float(), val_y.to(device).float()
                outputs = model(val_x)
                loss = criterion(val_x, val_y, outputs)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        min_val_loss = min(min_val_loss, val_loss)
        auc = eval_auc(model)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(val_loss)
        print(f'Epoch {epoch_id} validation loss: {val_loss}, Test AUC: {auc}')

    return min_val_loss


class BinaryClassificationPerformance:

    def __init__(self):
        self.y_true = []  # binary labels, 0 Negative, 1 Positive
        self.y_logit = []  # logit values, the possibility of being positive

    def add(self, y_true, y_logit):
        self.y_true.append(y_true)
        self.y_logit.append(y_logit)

    def get_accuracy(self, threshold=0.5):
        y_pred = [1 if p > threshold else 0 for p in self.y_logit]
        return accuracy_score(self.y_true, y_pred)

    def get_recall(self, threshold=0.5):
        y_pred = [1 if p > threshold else 0 for p in self.y_logit]
        return recall_score(self.y_true, y_pred, zero_division=0)

    def get_precision(self, threshold=0.5):
        y_pred = [1 if p > threshold else 0 for p in self.y_logit]
        return precision_score(self.y_true, y_pred, zero_division=0)

    def get_f1(self, threshold=0.5):
        y_pred = [1 if p > threshold else 0 for p in self.y_logit]
        return f1_score(self.y_true, y_pred, zero_division=0)

    def get_auc(self):
        return roc_auc_score(self.y_true, self.y_logit)

    def get_best_threshold(self, method='youden_index'):
        if method == 'youden_index':
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_logit)
            youden_index = tpr - fpr
            best_threshold = thresholds[np.argmax(youden_index)]
            return best_threshold
        elif method == 'highest_f1':
            best_threshold = 0
            best_f1 = 0
            for threshold in np.arange(0, 1, 0.001):
                y_pred = [1 if p > threshold else 0 for p in self.y_logit]
                f1 = f1_score(self.y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            return best_threshold
        else:
            raise ValueError(f'Unknown method: {method}, available methods are youden_index and highest_f1')

    def get_threshold_by_precision(self, precision):
        # 找到满足precision的最优threshold
        best_threshold = 0
        best_f1 = 0
        for threshold in np.arange(0, 1, 0.001):
            y_pred = [1 if p > threshold else 0 for p in self.y_logit]
            p = precision_score(self.y_true, y_pred)
            if p >= precision:
                f1 = f1_score(self.y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        return best_threshold