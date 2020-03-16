import torch
from torch import nn

conv_activ = nn.LeakyReLU
signal_network = nn.Sequential(
    nn.Conv1d(3, 6, 64),
    nn.MaxPool1d(8),
    conv_activ(),
    nn.Conv1d(6, 12, 32),
    nn.MaxPool1d(4),
    conv_activ(),
    nn.Conv1d(12, 18, 16),
    nn.MaxPool1d(4),
    conv_activ(),
    nn.Conv1d(18, 24, 16),
    nn.MaxPool1d(2),
    conv_activ(),
    nn.Flatten()
)
signal_out_shape = signal_network(torch.randn(1, 3, 6000)).shape
signal_network.cuda()
# print(signal_out_shape)

pars_activ = lambda: nn.LeakyReLU(negative_slope=0.05)
pars_network = nn.Sequential(
    nn.Linear(12, 36),
    pars_activ(),
    nn.Linear(36, 64),
    pars_activ(),
    nn.Linear(64, 96),
    pars_activ(),
    nn.Linear(96, 128),
    pars_activ()
)
pars_out_shape = pars_network(torch.randn(1, 12)).shape
pars_network.cuda()
# print(pars_out_shape)

final_activ = nn.Sigmoid
final_network = nn.Sequential(
    nn.Linear(signal_out_shape[1]+pars_out_shape[1], 128),
    nn.Dropout(p=0.05),
    final_activ(),
    nn.Linear(128, 32),
    nn.Dropout(p=0.05),
    final_activ(),
    nn.Linear(32, 7),
    nn.Dropout(p=0.05),
    nn.Softmax(dim=1)
)
final_network.cuda()
# print(final_network(torch.randn(1, signal_out_shape[1]+pars_out_shape[1])).shape)
