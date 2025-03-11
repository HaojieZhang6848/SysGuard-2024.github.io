import pandas as pd
import torch.optim
import numpy as np
import random
from torch.utils.data import DataLoader
from lstm_utils import load_device_lstm_inout_feature_names
from device_utils import device_id_to_name, state_extremas
from device_utils import get_device_type
from collections import defaultdict
from ml_utils import DeviceCheckerDataset, BinaryClassificationPerformance
from lstm_train import LSTM_Model, n_in_features, n_out_features, device, device_lstm_inout_feature_names, device_names, device_id_to_name
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device_lstm_inout_feature_names = load_device_lstm_inout_feature_names()

device_id_to_n_valid_out_features = {
    device_id: len(device_lstm_inout_feature_names[device_id_to_name[device_id]]['out']) for device_id in device_id_to_name.keys()
}

device_id_to_diff_normalization_factor = defaultdict(float)
for device_id in device_id_to_name.keys():
    out_features = device_lstm_inout_feature_names[device_id_to_name[device_id]]['out']
    for of in out_features:
        _, feature_type = of.split('_')
        if feature_type in state_extremas:
            device_id_to_diff_normalization_factor[device_id] += state_extremas[feature_type][1]
        else:
            print(f'feature_type: {feature_type} not found in state_extremas')
            exit(1)


lstm_test_x = np.load('data-dc/lstm_test_x.npy')
lstm_test_y_pgm = np.load('data-dc/lstm_test_y_pgm.npy')
lstm_test_y_actual = np.load('data-dc/lstm_test_y_actual.npy')
lstm_test_labels = np.load('data-dc/lstm_test_labels.npy')
lstm_test_on_off = np.load('data-dc/lstm_test_on_off.npy')

dc_dataset = DeviceCheckerDataset(lstm_test_x, lstm_test_y_pgm, lstm_test_y_actual, lstm_test_labels, lstm_test_on_off)
dc_dataloader = DataLoader(dc_dataset, batch_size=1, shuffle=False)


def eval_stage():
    n_hidden_dim = 106
    num_lstm_layers = 2
    lstm_model = LSTM_Model(n_in_features, n_hidden_dim, n_out_features, num_lstm_layers=num_lstm_layers, dropout_rate=0.5).to(device)
    lstm_model.load_state_dict(torch.load('data-dc/lstm_model_final.pth'))
    lstm_model.eval()

    # diffs[<device_name>][<normal/abnormal>][<on/off] = list of diffs
    diffs = dict()

    # device_type_y_true[<device_type>] = list of y_true, device_type_y_pred[<device_type>] = list of y_pred
    device_type_y_true = defaultdict(list)
    device_type_y_logits = defaultdict(list)

    overall_performance = BinaryClassificationPerformance()
    device_type_performance = defaultdict(BinaryClassificationPerformance)

    for x, y_pgm, y_actual, label, on_off in tqdm(dc_dataloader):
        x = x.to(device).float()
        y_pgm = y_pgm.to(device).float()
        d_id = int(x[0, 0, 0].item())
        d_name = device_id_to_name[d_id]
        d_type = get_device_type(d_name)
        if d_name not in diffs:
            diffs[d_name] = dict()
            diffs[d_name]['Normal'] = dict()
            diffs[d_name]['Abnormal'] = dict()
            diffs[d_name]['Normal']['On'] = []
            diffs[d_name]['Normal']['Off'] = []
            diffs[d_name]['Abnormal']['On'] = []
            diffs[d_name]['Abnormal']['Off'] = []
            print(f'processing device: {d_name}')
        with torch.no_grad():
            y_lstm = lstm_model(x)
        y_total = y_pgm + y_lstm
        y_total = y_total.cpu().detach().numpy()
        y_actual = y_actual.cpu().detach().numpy()
        label = 'Abnormal' if label[0].item() == 1 else 'Normal'

        valid_n_out_features = device_id_to_n_valid_out_features[d_id]
        y_total[:, valid_n_out_features:] = 0
        y_actual[:, valid_n_out_features:] = 0
        this_diff = np.abs(y_total[0] - y_actual[0]).sum()
        nf = device_id_to_diff_normalization_factor[d_id]
        this_diff /= nf
        onoff = 'On' if on_off[0].item() == 1 else 'Off'

        diffs[d_name][label][onoff].append(this_diff)
        device_type_y_true[d_type].append(1 if label == 'Abnormal' else 0)
        device_type_y_logits[d_type].append(this_diff)

        y_true = 1 if label == 'Abnormal' else 0
        y_logit = this_diff
        overall_performance.add(y_true, y_logit)
        device_type_performance[d_type].add(y_true, y_logit)
        
    out_table = pd.DataFrame(columns=['Device_Type', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
        
    best_threshold = overall_performance.get_best_threshold(method='youden_index')
    accuracy = overall_performance.get_accuracy(best_threshold)
    precision = overall_performance.get_precision(best_threshold)
    recall = overall_performance.get_recall(best_threshold)
    f1 = overall_performance.get_f1(best_threshold)
    auc = overall_performance.get_auc()
    print(f'Overall, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}')
    out_table.loc[len(out_table)] = ['Overall', accuracy, precision, recall, f1, auc]
    
    
    for device_type in device_type_performance:
        accuracy = device_type_performance[device_type].get_accuracy(best_threshold)
        precision = device_type_performance[device_type].get_precision(best_threshold)
        recall = device_type_performance[device_type].get_recall(best_threshold)
        f1 = device_type_performance[device_type].get_f1(best_threshold)
        auc = device_type_performance[device_type].get_auc()
        print(f'device_type: {device_type}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}')
        out_table.loc[len(out_table)] = [device_type, accuracy, precision, recall, f1, 0]
    
    # save the results
    out_table.to_csv('data-dc/evaluation_results_lstm.csv', index=False)
    
    # optimum_threshold_1 = overall_performance.get_best_threshold(method='highest_f1')
    # optimum_threshold_2 = overall_performance.get_best_threshold(method='youden_index')
    # accuracy_1 = overall_performance.get_accuracy(optimum_threshold_1)
    # accuracy_2 = overall_performance.get_accuracy(optimum_threshold_2)
    # precision_1 = overall_performance.get_precision(optimum_threshold_1)
    # precision_2 = overall_performance.get_precision(optimum_threshold_2)
    # recall_1 = overall_performance.get_recall(optimum_threshold_1)
    # recall_2 = overall_performance.get_recall(optimum_threshold_2)
    # f1_1 = overall_performance.get_f1(optimum_threshold_1)
    # f1_2 = overall_performance.get_f1(optimum_threshold_2)
    # auc = overall_performance.get_auc()

    # print(
    #     f'When using highest f1 threshold: Accuracy: {accuracy_1:.3f}, Precision: {precision_1:.3f}, Recall: {recall_1:.3f}, F1: {f1_1:.3f}, AUC: {auc:.3f}, Best Threshold: {optimum_threshold_1:.3f}')
    # print(
    #     f'When using youden index threshold: Accuracy: {accuracy_2:.3f}, Precision: {precision_2:.3f}, Recall: {recall_2:.3f}, F1: {f1_2:.3f}, AUC: {auc:.3f}, Best Threshold: {optimum_threshold_2:.3f}')


def eval_auc(lstm_model):
    lstm_model.eval()
    # diffs[<device_name>][<normal/abnormal>][<on/off] = list of diffs
    diffs = dict()

    all_y_true = []
    all_y_logit = []

    for x, y_pgm, y_actual, label, on_off in tqdm(dc_dataloader):
        x = x.to(device).float()
        y_pgm = y_pgm.to(device).float()
        d_id = int(x[0, 0, 0].item())
        d_name = device_id_to_name[d_id]
        if d_name not in diffs:
            diffs[d_name] = dict()
            diffs[d_name]['Normal'] = dict()
            diffs[d_name]['Abnormal'] = dict()
            diffs[d_name]['Normal']['On'] = []
            diffs[d_name]['Normal']['Off'] = []
            diffs[d_name]['Abnormal']['On'] = []
            diffs[d_name]['Abnormal']['Off'] = []
        with torch.no_grad():
            y_lstm = lstm_model(x)
        y_total = y_pgm + y_lstm
        y_total = y_total.cpu().detach().numpy()
        y_actual = y_actual.cpu().detach().numpy()
        label = 'Abnormal' if label[0].item() == 1 else 'Normal'

        valid_n_out_features = device_id_to_n_valid_out_features[d_id]
        y_total[:, valid_n_out_features:] = 0
        y_actual[:, valid_n_out_features:] = 0
        this_diff = np.abs(y_total[0] - y_actual[0]).sum()
        nf = device_id_to_diff_normalization_factor[d_id]
        this_diff /= nf

        onoff = 'On' if on_off[0].item() == 1 else 'Off'
        diffs[d_name][label][onoff].append(this_diff)

        y_actual = 1 if label == 'Abnormal' else 0
        y_logit = this_diff
        all_y_true.append(y_actual)
        all_y_logit.append(y_logit)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_y_true, all_y_logit)
    return auc


if __name__ == '__main__':
    eval_stage()
