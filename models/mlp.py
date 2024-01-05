import torch
import torch.nn as nn
# model= nn.Sequential(
#     nn.Linear(4, 64),
#     nn.ReLU(),
#     nn.Linear(64, 16),
#     nn.ReLU(),
#     nn.Linear(16, 4),
#     nn.ReLU(),
#     nn.Linear(4,1),
# )

model= nn.Sequential(
    nn.Linear(4, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256,64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 4),
    nn.ReLU(),
    nn.Linear(4,1),
)


def load_model(state='../pt_files/mlp_video1.pt', device='cpu'):
    model.to(device)
    model.load_state_dict(torch.load(state, map_location=torch.device(device)))
    model.eval().to(device)
    return model

'''
measure size of model
'''
device = 'cuda'
from torchsummary import summary
from thop import profile
# print(summary(model,(1,1,4)))
flops, params = profile(model.to(device), inputs=(torch.rand(1,10,4).to(device),))
print(f'Flops: {flops}, Parameters:{params}')
gflops = flops / 1e9
print(f"GFLOPs: {gflops:.2f}")
# print(stat(model,(1,1,4)))
