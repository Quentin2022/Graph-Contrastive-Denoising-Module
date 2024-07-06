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
        # Graph2Vec
        idx = torch.triu_indices(self.num_nodes, self.num_nodes, 1)
        x = adj[:, idx[0], idx[1]]
        x = self.mlp(x)

        # Reconstruction Graph
        adj_new = torch.zeros_like(adj)
        adj_new[:, idx[0], idx[1]] = x
        adj_new[:, idx[1], idx[0]] = x

        return adj_new

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, ifTransAdj, TransAdj, 
                 embedding_hidden_layers, embedding_initial_weights, embedding_if_finetune):
        super(GraphConvolution, self).__init__()
        self.ifTransAdj = ifTransAdj
        self.TransAdj = TransAdj
        self.num_nodes = num_nodes
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.graph_embedding = False
        if ifTransAdj:
            self.graph_embedding = GraphEmbedding(num_nodes, embedding_hidden_layers, 
                                                  embedding_initial_weights, embedding_if_finetune)

    def forward(self, input, adj):
        if self.graph_embedding:
            adj = self.graph_embedding(adj)

        if self.TransAdj == 'L':
            adj = self.compute_laplacian(adj)
        elif self.TransAdj == 'norL':
            adj = self.compute_normalized_laplacian(adj)
        
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return output

    def compute_laplacian(self, adj):
        degree_matrix = torch.sum(adj, dim=-1).diag_embed()
        laplacian = degree_matrix - adj
        return laplacian

    def compute_normalized_laplacian(self, adj):
        degree_matrix = torch.sum(adj, dim=-1)
        degree_inv_sqrt = torch.pow(degree_matrix, -0.5).diag_embed()
        laplacian = torch.eye(self.num_nodes, device=adj.device) - torch.matmul(degree_inv_sqrt, torch.matmul(adj, degree_inv_sqrt))
        return laplacian

class GCN(nn.Module):
    def __init__(self, layer_sizes, dropout=0.5, readout='mean', encode_dim=32, num_nodes=16, ifTransAdj=True, TransAdj='A',
                 embedding_hidden_layers=None, embedding_initial_weights=None, embedding_if_finetune=True):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.encode_dim = encode_dim
        for i in range(len(layer_sizes) - 1):
            self.layers.append(GraphConvolution(layer_sizes[i], layer_sizes[i + 1], num_nodes, ifTransAdj, TransAdj,
                                                embedding_hidden_layers, embedding_initial_weights, embedding_if_finetune))
        self.encode_layer = nn.Linear(layer_sizes[-1] * num_nodes, encode_dim)

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.encode_layer(x)
        return x


class WeightedGraphDecoder(nn.Module):
    def __init__(self, input_dim, num_nodes):
        super(WeightedGraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_nodes * num_nodes)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        adj_reconstructed = x.view(-1, self.num_nodes, self.num_nodes)
        adj_reconstructed = (adj_reconstructed + adj_reconstructed.transpose(1, 2)) / 2
        return adj_reconstructed


class GAE(nn.Module):
    def __init__(self, encoder, num_nodes):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = WeightedGraphDecoder(encoder.encode_dim, num_nodes)

    def forward(self, features, adj):
        z = self.encoder(features, adj)
        reconstructed_adj = self.decoder(z)
        return reconstructed_adj
