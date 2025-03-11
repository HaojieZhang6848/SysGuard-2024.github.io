from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import torch
import numpy as np

class ModelPerformance:
    labels_true = []  # true label, 0 or 1
    labels_pred = []  # predicted label, 0 or 1, for accuracy, recall, precision
    labels_pred_prob = []  # probability of being 1, for AUC

    def __init__(self):
        self.labels_true = []
        self.labels_pred = []
        self.labels_pred_prob = []

    def add(self, label_true, label_pred, label_pred_prob):
        self.labels_true.append(label_true)
        self.labels_pred.append(label_pred)
        self.labels_pred_prob.append(label_pred_prob)

    def get_accuracy(self):
        return accuracy_score(self.labels_true, self.labels_pred)

    def get_recall(self):
        return recall_score(self.labels_true, self.labels_pred, zero_division=0)

    def get_precision(self):
        return precision_score(self.labels_true, self.labels_pred, zero_division=0)

    def get_f1(self):
        return f1_score(self.labels_true, self.labels_pred, zero_division=0)

    def get_auc(self):
        return roc_auc_score(self.labels_true, self.labels_pred_prob)

    def get_best_threshold(self):
        best_threshold = 0
        best_f1 = 0
        for threshold in np.arange(0, 1, 0.001):
            labels_pred = [1 if p > threshold else 0 for p in self.labels_pred_prob]
            f1 = f1_score(self.labels_true, labels_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        return best_threshold


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

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
        torch.save(model.state_dict(), 'gnn_checkpoint.pt')
        self.val_loss_min = val_loss
