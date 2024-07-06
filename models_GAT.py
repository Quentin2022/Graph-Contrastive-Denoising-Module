import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphEmbedding(nn.Module):
    def __init__(self, num_nodes, embeddings_hidden_layers, embeddings_initial_weights, embeddings_if_finetune):
        super(GraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        layers = []
        input_dim = num_nodes * (num_nodes - 1) // 2
        for i, output_dim in enumerate(embeddings_hidden_layers):
            linear_layer = nn.Linear(input_dim, output_dim)
            # If initial weights are provided, use them to initialise the
            if embeddings_initial_weights:
                linear_layer.weight.data.copy_(embeddings_initial_weights[i*2])
                linear_layer.bias.data.copy_(embeddings_initial_weights[i*2 + 1])
            linear_layer.weight.requires_grad = embeddings_if_finetune
            linear_layer.bias.requires_grad = embeddings_if_finetune
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            input_dim = output_dim
        output_layer = nn.Linear(input_dim, num_nodes * (num_nodes - 1) // 2)
        if embeddings_initial_weights:
            output_layer.weight.data.copy_(embeddings_initial_weights[-2])
            output_layer.bias.data.copy_(embeddings_initial_weights[-1])
        output_layer.weight.requires_grad = embeddings_if_finetune
        output_layer.bias.requires_grad = embeddings_if_finetune
        layers.append(output_layer)
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, adj):
        idx = torch.triu_indices(self.num_nodes, self.num_nodes, 1)
        x = adj[:, idx[0], idx[1]]
        x = self.mlp(x)
        adj_new = torch.zeros_like(adj)
        adj_new[:, idx[0], idx[1]] = x
        adj_new[:, idx[1], idx[0]] = x
        return adj_new


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, num_heads, ifTransAdj, TransAdj, embedding_hidden_layers, embedding_initial_weights, embedding_if_finetune, neighbor_rate):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.neighbor_rate = neighbor_rate
        self.head_dim = output_dim // num_heads
        
        self.W = nn.ParameterList([nn.Parameter(torch.empty(input_dim, self.head_dim)) for _ in range(num_heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.empty(2 * self.head_dim, 1)) for _ in range(num_heads)])
        for i in range(num_heads):
            nn.init.kaiming_uniform_(self.W[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.a[i], a=math.sqrt(5))

        self.ifTransAdj = ifTransAdj
        if ifTransAdj:
            self.graph_embedding = GraphEmbedding(num_nodes, embedding_hidden_layers, embedding_initial_weights, embedding_if_finetune)

    def forward(self, h, adj):
        if self.ifTransAdj:
            adj = self.graph_embedding(adj)

        batch_size, num_nodes = h.size(0), h.size(1)
        h_prime_list = []

        for k in range(self.num_heads):
            Wh = torch.matmul(h, self.W[k])

            Wh_i = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            Wh_j = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)

            e = torch.cat([Wh_i, Wh_j], dim=-1)
            e = torch.matmul(e, self.a[k]).squeeze(-1)
            e = F.leaky_relu(e, 0.2)

            attention_scores = e.clone()
            num_neighbors = int(self.neighbor_rate * num_nodes)
            topk_values, topk_indices = torch.topk(attention_scores, num_neighbors, dim=2)
            mask = torch.zeros_like(e).scatter_(2, topk_indices, 1)
            masked_attention = e.masked_fill(mask == 0, -1e9)
            attention = F.softmax(masked_attention, dim=2)

            attention = attention * adj

            h_prime = torch.matmul(attention, Wh)
            h_prime_list.append(h_prime)

        h_prime = torch.cat(h_prime_list, dim=-1)  # Concatenate along the feature dimension
        return h_prime


class GAT(nn.Module):
    def __init__(self, layer_sizes, dropout=0.5, readout='mean', num_classes=2, num_nodes=16, num_heads=2, 
                 ifTransAdj=True, TransAdj='A', embedding_hidden_layers=None, embedding_initial_weights=None, 
                 embedding_if_finetune=True, neighbor_rate=0.2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.readout = readout
        self.dropout = dropout
        self.ifTransAdj = ifTransAdj
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(GraphAttentionLayer(layer_sizes[i], layer_sizes[i + 1], num_nodes, num_heads, 
                                                   ifTransAdj, TransAdj, embedding_hidden_layers, 
                                                   embedding_initial_weights, embedding_if_finetune, neighbor_rate))

        if readout == 'fc':
            self.fc_pool = nn.Linear(num_nodes * layer_sizes[-1], num_classes)

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.layers[-1](x, adj)

        if self.readout == 'mean':
            x = torch.mean(x, dim=1)
        elif self.readout == 'sum':
            x = torch.sum(x, dim=1)
        elif self.readout == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.readout == 'fc':
            x = x.view(x.size(0), -1)
            x = self.fc_pool(x)

        # x = F.log_softmax(x, dim=1)
        return x
