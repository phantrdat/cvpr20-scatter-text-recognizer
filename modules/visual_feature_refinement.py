import torch
import torch.nn as nn
class Refinement(nn.Module):
    def __init__(self, input_size, output_size):
        super(Refinement, self).__init__()
        self.iid = nn.Linear(input_size, input_size)
        self.decoder = nn.Linear(input_size, output_size)
    def forward(self, input):
        x = self.iid(input)
        x = self.decoder(x)
        return x