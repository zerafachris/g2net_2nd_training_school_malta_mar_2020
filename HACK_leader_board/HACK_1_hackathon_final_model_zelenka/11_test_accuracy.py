import numpy as np
import torch
from torch import nn
import pandas as pd

from network import *

def dataset_tensors(df, dtype=torch.float32, classes=7):
    parameter_keys = ('receiver_latitude', 'receiver_longitude', 'receiver_elevation_m', 'p_arrival_sample', 'p_travel_sec', 's_arrival_sample', 'source_latitude', 'source_longitude', 'source_depth_km', 'snr_db_E', 'snr_db_N', 'snr_db_Z')
    signal_keys = ('E', 'N', 'Z')
    class_key = 'target'
    parameter_tensor = torch.from_numpy(np.stack([df[key].values for key in parameter_keys], axis=-1)).to(dtype=dtype)
    signal_tensor = torch.from_numpy(np.stack([np.stack(df[key].values, axis=0) for key in signal_keys], axis=1)).to(dtype=dtype)
    if not classes is None:
        labels = df[class_key].values
        labels_tensor = torch.zeros(len(labels), classes, dtype=dtype)
    #     print(len(labels), classes)
        for i in range(len(labels)):
    #         print(i, labels[i])
            labels_tensor[i][labels[i]] = 1.
        return parameter_tensor, signal_tensor, labels_tensor
    else:
        return parameter_tensor, signal_tensor

df_test = pd.read_pickle('../data/df_test.pkl.gzip', compression='gzip')
test_pars, test_signals, test_labels = dataset_tensors(df_test)

#test_pars = test_pars.cuda()
#test_signals = test_signals.cuda()
#test_labels = test_labels.cuda()

signal_network.cpu()
pars_network.cpu()
final_network.cpu()

signal_network.load_state_dict(torch.load('signal_state.pt'))
pars_network.load_state_dict(torch.load('pars_state.pt'))
final_network.load_state_dict(torch.load('final_state.pt'))

signal_network.eval()
pars_network.eval()
final_network.eval()

meanproduct1 = signal_network(test_signals)
meanproduct2 = pars_network(test_pars)
meanproduct3 = torch.cat((meanproduct1, meanproduct2), axis=1)
test_out = final_network(meanproduct3)

correct = 0
for i in range(len(test_out)):
    if test_out[i].cpu().detach().numpy().argmax()==test_labels[i].cpu().detach().numpy().argmax():
        correct += 1
print(correct/len(test_out))
