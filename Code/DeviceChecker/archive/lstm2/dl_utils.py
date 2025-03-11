import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

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
        torch.save(model.state_dict(), 'data-dc/lstm2_model.pth')
        self.val_loss_min = val_loss


class Lstm2Dataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


def train(model, optimizer, scheduler, criterion, train_loader, val_loader, epoch, early_stopping, device):
    highest_f1 = 0
    highest_auc = 0
    for epoch_id in range(1, epoch + 1):
        model.train()
        train_loss = 0
        for x, labels in tqdm(train_loader):
            x, labels = x.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_true = []
        val_pred = []
        val_logits = []
        val_loss = 0
        with torch.no_grad():
            for x, labels in tqdm(val_loader):
                x, labels = x.to(device).float(), labels.to(device).long()
                out = model(x) # [batch_size, 2]
                loss = criterion(out, labels)
                val_loss += loss.item()
                val_true.extend(labels.cpu().numpy())
                y_pred = out.argmax(dim=1)
                val_pred.extend(y_pred.cpu().numpy())
                y_logits = out.softmax(dim=1)
                val_logits.extend(y_logits[:, 1].cpu().numpy())
                
            val_loss /= len(val_loader)

        f1 = f1_score(val_true, val_pred)
        if f1 > highest_f1:
            highest_f1 = f1
        auc = roc_auc_score(val_true, val_logits)
        if auc > highest_auc:
            highest_auc = auc
        
        scheduler.step(-auc)
        early_stopping(-auc, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

        print(f'Epoch {epoch_id}, Train Loss: {train_loss}, Val Loss: {val_loss}, F1: {f1}, Auc: {auc}')

    return highest_auc