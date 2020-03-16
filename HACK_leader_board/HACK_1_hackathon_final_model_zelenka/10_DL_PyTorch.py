import numpy as np
# from numpy import fft as npfft
import pandas as pd
import matplotlib.pyplot as plt

from network import *

import torch
from torch import nn, optim

# When loading be sure to define the compression
df_train = pd.read_pickle('../data/df_train.pkl.gzip', compression='gzip')
df_test = pd.read_pickle('../data/df_test.pkl.gzip', compression='gzip')

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

train_pars, train_signals, train_labels = dataset_tensors(df_train)
test_pars, test_signals, test_labels = dataset_tensors(df_test)

train_pars = train_pars.cuda()
train_signals = train_signals.cuda()
train_labels = train_labels.cuda()

test_pars = test_pars.cuda()
test_signals = test_signals.cuda()
test_labels = test_labels.cuda()

del df_test
del df_train

# signal_network = nn.Sequential(
#     nn.Conv1d(3, 6, 64),
#     nn.MaxPool1d(6),
#     nn.ReLU(),
#     nn.Conv1d(6, 12, 32),
#     nn.MaxPool1d(8),
#     nn.ReLU(),
#     nn.Conv1d(12, 16, 32),
#     nn.MaxPool1d(10),
#     nn.ReLU(),
#     nn.Flatten()
# )
# signal_out_shape = signal_network(torch.randn(1, 3, 6000)).shape
# signal_network.cuda()
# # print(signal_out_shape)

# pars_network = nn.Sequential(
#     nn.Linear(12, 36),
#     nn.Sigmoid(),
#     nn.Linear(36, 64),
#     nn.Sigmoid()
# )
# pars_out_shape = pars_network(torch.randn(1, 12)).shape
# pars_network.cuda()
# # print(pars_out_shape)

# final_network = nn.Sequential(
#     nn.Linear(signal_out_shape[1]+pars_out_shape[1], 128),
#     nn.Sigmoid(),
#     nn.Linear(128, 32),
#     nn.Sigmoid(),
#     nn.Linear(32, 7),
#     nn.Softmax(dim=1)
# )
# final_network.cuda()
# print(final_network(torch.randn(1, signal_out_shape[1]+pars_out_shape[1])).shape)

torch.save(signal_network.state_dict(), 'initial_signal_state.pt')
torch.save(pars_network.state_dict(), 'initial_pars_state.pt')
torch.save(final_network.state_dict(), 'initial_final_state.pt')

crit = nn.BCELoss()
opt = optim.Adam([{'params': signal_network.parameters()},
    {'params': pars_network.parameters()},
    {'params': final_network.parameters()}], lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(opt, 100, gamma=.1**(1/4))
epochs = 1000
batch_size = 32
dslen = len(train_pars)
chunks_num = dslen//batch_size + (0 if dslen%batch_size==0 else 1)

train_signals_chunks = train_signals.chunk(chunks_num)
train_pars_chunks = train_pars.chunk(chunks_num)
train_labels_chunks = train_labels.chunk(chunks_num)


best_loss = 100.
train_losses = []
test_losses = []
for e in range(1, epochs+1):
    train_loss = 0.
    pars_network.train()
    signal_network.train()
    final_network.train()
    for chunk in range(chunks_num):
        # print('starting batch %04i' % chunk)
        opt.zero_grad()
        # print(torch.norm(final_network[0]._parameters['weight'].grad))
        meanproduct1 = signal_network(train_signals_chunks[chunk])
        meanproduct2 = pars_network(train_pars_chunks[chunk])
        meanproduct3 = torch.cat((meanproduct1, meanproduct2), axis=1)
        train_out = final_network(meanproduct3)
        batch_loss = crit(train_out, train_labels_chunks[chunk])
        batch_loss.backward()
        # print(torch.norm(final_network[0]._parameters['weight'].grad))
        opt.step()
        train_loss += batch_loss
    train_loss /= chunks_num
    with torch.no_grad():
        pars_network.eval()
        signal_network.eval()
        final_network.eval()
        meanproduct1 = signal_network(test_signals)
        meanproduct2 = pars_network(test_pars)
        meanproduct3 = torch.cat((meanproduct1, meanproduct2), axis=1)
        test_out = final_network(meanproduct3)
        test_loss = crit(test_out, test_labels)
    if test_loss<best_loss:
        torch.save(signal_network.state_dict(), 'signal_state.pt')
        torch.save(pars_network.state_dict(), 'pars_state.pt')
        torch.save(final_network.state_dict(), 'final_state.pt')
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print('%i    %f    %f' % (e, train_loss, test_loss), flush=True)
    scheduler.step()

correct = 0
for i in range(len(test_out)):
    if test_out[i].cpu().detach().numpy().argmax()==test_labels[i].cpu().detach().numpy().argmax():
        correct += 1
print('# final test accuracy %f' % (correct/len(test_out)), flush=True)
