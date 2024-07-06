import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(GraphEmbedding, self).__init__()
        layers = []
        current_dim = input_dim
        for output_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = output_dim
        layers.append(nn.Linear(hidden_layers[-1], input_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def compute_normalized_laplacian(adj, num_nodes):
    degree_matrix = torch.sum(adj, dim=-1)
    degree_inv_sqrt = torch.pow(degree_matrix, -0.5).diag_embed()
    laplacian = torch.eye(num_nodes, device=adj.device) - torch.matmul(degree_inv_sqrt, torch.matmul(adj, degree_inv_sqrt))
    return laplacian

def rbf_kernel_matrix(eigenv, gamma=10.0):
    sum_x = torch.sum(eigenv ** 2, dim=1, keepdim=True)  # [batch_size, 1]
    sum_y = sum_x.view(1, -1)  # [1, batch_size]
    dist_sq = sum_x + sum_y - 2 * torch.matmul(eigenv, eigenv.t())  # [batch_size, batch_size]

    K = torch.exp(-gamma * dist_sq)

    K = torch.triu(K) + torch.triu(K, 1).t()
    return K

def rbf_matrix(eigenv, device='cuda'):
    # batch_size, d = eigenv.shape

    sigma = eigenv.std(dim=0, unbiased=False)  # [d]
    gamma = 1 / (4 * sigma ** 2)  # [d]

    dist_sq = (eigenv.unsqueeze(1) - eigenv.unsqueeze(0)) ** 2  # [batch_size, batch_size, d]

    gamma_matrix = gamma.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
    weighted_dist_sq = dist_sq * gamma_matrix  # [batch_size, batch_size, d]

    exp_component = torch.exp(-weighted_dist_sq)  # [batch_size, batch_size, d]

    K = exp_component.mean(dim=-1)  # [batch_size, batch_size]

    K = torch.triu(K) + torch.triu(K, 1).t()
    return K

def contrastive_loss(embeddings, labels, num_nodes, metric, loss_type, margin, gamma):
    batch_size = embeddings.size(0)
    labels_expanded = labels.unsqueeze(0)

    # Reconstruction Graph
    idx = torch.triu_indices(num_nodes, num_nodes, 1)
    adj = torch.zeros(batch_size, num_nodes, num_nodes).to(device)
    adj[:, idx[0], idx[1]] = embeddings
    adj[:, idx[1], idx[0]] = embeddings

    L = compute_normalized_laplacian(adj, num_nodes)
    eigenvalues = torch.linalg.eigvalsh(L)
    eigenvalues.sort()
    eigenv = eigenvalues[:, 1:]
    # nor_eigen = (eigenv - eigenv.mean(dim=0, keepdim=True)) / eigenv.std(dim=0, keepdim=True) + 1e-6 
    
    similarity_global = rbf_matrix(eigenv)
    
    # Compute distance based on the selected metric
    if metric == 'CosSimilarity':
        similarity_local = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    elif metric == 'L2':
        similarity_local = torch.cdist(embeddings, embeddings, p=2) / embeddings.size(1)
    elif metric == 'L1':
        similarity_local = torch.cdist(embeddings, embeddings, p=1) / embeddings.size(1)
    
    similarity = similarity_global * gamma + similarity_local

    # Construct masks for positive and negative samples
    pos_mask = (labels_expanded == labels_expanded.T) & ~torch.eye(batch_size, device=embeddings.device).bool()
    neg_mask = (labels_expanded != labels_expanded.T)

    # pos_similarity = similarity[pos_mask].unsqueeze(1)
    # neg_similarity = similarity[neg_mask].unsqueeze(1)

    # Compute loss based on the selected loss type
    if loss_type == 'MarginBased':
        # Margin-based contrastive loss
        loss = pos_similarity + F.relu(margin - neg_similarity).mean()
    elif loss_type == 'TripletLoss':
        loss = F.relu(pos_similarity - neg_similarity + margin).mean()
    elif loss_type == 'TripLoss':
        loss = F.relu(pos_similarity - 0.01).mean() + F.relu(0.3 - neg_similarity).mean()

    elif loss_type == 'InfoNCELoss':
        similarity = similarity / 0.1
        pos_sim = similarity * pos_mask
        pos_exp_sum = torch.exp(pos_sim).sum(dim=1)
        all_exp_sum = torch.exp(similarity).sum(dim=1)
        loss = -torch.log(pos_exp_sum / all_exp_sum)
        loss = loss.mean()

    return loss


def GCD(train_loader, test_loader, num_nodes, epochs=30, hidden_layers=[240, 240], 
                            dropout_rate=0.5, sim_metric='CosSimilarity', margin=0.3, pretraining_lr=0.001, 
                            pretraining_weight_decay=0, embeddings_loss_type='TripLoss', if_plotadj=False, gamma=0.0001):
    input_dim = num_nodes * (num_nodes - 1) // 2  # Calculate the input dimension based on the upper triangle of the adjacency matrix
    model = GraphEmbedding(input_dim, hidden_layers, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=pretraining_lr, weight_decay=pretraining_weight_decay)
    train_losses, test_losses = [], []

    # Graph2Vec index
    idx = torch.triu_indices(num_nodes, num_nodes, 1)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for _, labels, graphs in train_loader:
            labels = labels.to(device)
            graphs = graphs.to(device)
            # Graph2Vec
            graphs_flattened = graphs[:, idx[0], idx[1]]  # Flatten the upper triangle of the adjacency matrices
            embeddings = model(graphs_flattened)
            loss = contrastive_loss(embeddings, labels, num_nodes, metric=sim_metric, margin=margin, loss_type=embeddings_loss_type, gamma=gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Ploting
        embed_list = []
        label_list = []

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for _, labels, graphs in test_loader:
                labels = labels.to(device)
                graphs = graphs.to(device)
                graphs_flattened = graphs[:, idx[0], idx[1]]
                embeddings = model(graphs_flattened)

                embed_list.append(embeddings.clone())
                label_list.append(labels.clone())

                loss = contrastive_loss(embeddings, labels, num_nodes, metric=sim_metric, margin=margin, loss_type=embeddings_loss_type, gamma=gamma)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

        embed_plot = torch.cat(embed_list, dim=0).cpu()
        label_plot = torch.cat(label_list, dim=0).cpu()

    if if_plotadj:

        c = num_nodes

        adj = np.zeros((embed_plot.shape[0], c, c))
        triu_indices = np.triu_indices(c, 1)
        for i in range(embed_plot.shape[0]):
            adj[i][triu_indices] = embed_plot[i]
            adj[i] = adj[i] + adj[i].T

        class_0_indices = np.where(label_plot == 0)[0][:60]
        class_1_indices = np.where(label_plot == 1)[0][:60]

        fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(16, 9))
        for i, ax in enumerate(axes[0]):
            ax.imshow(adj[class_0_indices[i]], cmap='Greys', interpolation='nearest')
            ax.axis('off')
        for i, ax in enumerate(axes[1]):
            ax.imshow(adj[class_0_indices[i+10]], cmap='Greys', interpolation='nearest')
            ax.axis('off')
        for i, ax in enumerate(axes[2]):
            ax.imshow(adj[class_0_indices[i+20]], cmap='Greys', interpolation='nearest')
            ax.axis('off')
        for i, ax in enumerate(axes[3]):
            ax.imshow(adj[class_1_indices[i]], cmap='Greys', interpolation='nearest')
            ax.axis('off')
        for i, ax in enumerate(axes[4]):
            ax.imshow(adj[class_1_indices[i+10]], cmap='Greys', interpolation='nearest')
            ax.axis('off')
        for i, ax in enumerate(axes[5]):
            ax.imshow(adj[class_1_indices[i+20]], cmap='Greys', interpolation='nearest')
            ax.axis('off')

        plt.show()

        adj_pretrain = np.concatenate((adj[class_0_indices], adj[class_1_indices]), axis=0)
        # np.save('adj_pretrain.npy', adj_pretrain)

    model_params_list = []
    for name, param in model.state_dict().items():
        model_params_list.append(param)

    return train_losses, test_losses, model_params_list
