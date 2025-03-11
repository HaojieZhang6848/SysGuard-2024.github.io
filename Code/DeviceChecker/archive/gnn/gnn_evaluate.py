import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict
from dl_utils import ModelPerformance, EarlyStopping
from gnn_train_evaluate import GNN_Model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TAU = os.getenv('TAU', 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(f'data-dc/05_testing_preprocessed_gnn_{TAU}.pkl', 'rb') as f:
    testing_dataset = pickle.load(f)

bar = int(0.8 * len(testing_dataset))
validating_dataset = testing_dataset[bar:]
testing_dataset = testing_dataset[:bar]

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
            label_pred = 1 if label_pred_prob > 0.007 else 0

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
    hidden_dim = 110
    output_dim = 2
    model = GNN_Model(input_dim, hidden_dim, output_dim)
    model = model.to(device)
    # load state dict
    model.load_state_dict(torch.load('gnn_checkpoint.pt'))
    
    criterion = nn.CrossEntropyLoss()
    test(model, testing_dataset, criterion)



if __name__ == '__main__':
    main()
