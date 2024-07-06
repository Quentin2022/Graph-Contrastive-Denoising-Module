import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout=0.5, num_classes=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(prev_dim, layer_size))
            prev_dim = layer_size
        
        self.output_layer = nn.Linear(prev_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Flatten the input tensor if it comes with more than one dimension per sample
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        # return F.log_softmax(x, dim=1)
        return x
