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

# When loading be sure to define the compression
df_submission = pd.read_pickle('../data/df_submission.pkl.gzip', compression='gzip')
submit_pars, submit_signals = dataset_tensors(df_submission, classes=None)

signal_network.load_state_dict(torch.load('signal_state.pt'))
pars_network.load_state_dict(torch.load('pars_state.pt'))
final_network.load_state_dict(torch.load('final_state.pt'))

signal_network.eval()
pars_network.eval()
final_network.eval()

signal_network.cpu()
pars_network.cpu()
final_network.cpu()

meanproduct1 = signal_network(submit_signals)
meanproduct2 = pars_network(submit_pars)
meanproduct3 = torch.cat((meanproduct1, meanproduct2), axis=1)
submit_out = final_network(meanproduct3)

y_pred = [submit_out[i].detach().numpy().argmax() for i in range(df_submission.shape[0])]

df_submit = pd.DataFrame({'trace_id' : df_submission['trace_id'],
                          'submission' : y_pred})
print(df_submit.head())

df_submit.to_csv('./your_submission.csv',index=False)
