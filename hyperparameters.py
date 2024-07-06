import argparse

def parse_args_GCD():
    parser_GCD = argparse.ArgumentParser(description="GCD Pretraining Parameters")
    
    parser_GCD.add_argument('--pretraining_lr', type=float, default=0.001, help='Learning rate for pretraining')
    parser_GCD.add_argument('--pretraining_epochs', type=int, default=30, help='Number of epochs for pretraining')
    parser_GCD.add_argument('--embeddings_layers', type=int, nargs='+', default=[240, 240], help='List of hidden layer sizes for the embedding model')
    parser_GCD.add_argument('--pretraining_weight_decay', type=float, default=0, help='Weight decay for the optimizer')
    parser_GCD.add_argument('--embeddings_dropout_rate', type=float, default=0.5, help='Dropout rate for the embedding model')
    parser_GCD.add_argument('--sim_metric', type=str, default='CosSimilarity', help='Similarity metric to use (CosSimilarity, L2, L1)')
    parser_GCD.add_argument('--embeddings_loss_type', type=str, default='InfoNCELoss', help='Type of loss to use (MarginBased, TripletLoss, TripLoss)')
    parser_GCD.add_argument('--margin', type=float, default=0.3, help='Margin value for the loss function')
    parser_GCD.add_argument('--gamma', type=float, default=0.0001, help='Balance local and global features')
    parser_GCD.add_argument('--if_plotadj', type=bool, default=False, help='Whether to plot adjacency matrices')

    return parser_GCD.parse_args()


def get_gcn_params():
    return {
        'learning_rate': 0.001,
        'epochs': 50,
        'layer_sizes': [10, 16, 4],
        'weight_decay': 0.001,
        'dropout': 0.5,
        'readout': 'fc',
        'embedding_if_finetune': True,
        'finetune_lr': 0.0005,
        'smoothing_rate': 0.1,
        'gamma': 0.001
    }

def get_gin_params():
    return {
        'learning_rate': 0.001,
        'epochs': 50,
        'layer_sizes': [10, 16, 4],
        'weight_decay': 0.001,
        'dropout': 0.5,
        'readout': 'fc',
        'embedding_if_finetune': True,
        'finetune_lr': 0.0005,
        'smoothing_rate': 0.1,
        'gamma': 0.001
    }

def get_gat_params():
    return {
        'learning_rate': 0.005,
        'epochs': 50,
        'layer_sizes': [10, 16],
        'weight_decay': 0.0005,
        'dropout': 0.5,
        'readout': 'fc',
        'embedding_if_finetune': True,
        'finetune_lr': 0.0005,
        'smoothing_rate': 0.1,
        'Neighbor_rate': 0.5,  # rate=1 means global GAT
        'heads': 4,
        'gamma': 0.001
    }


def get_model_params(gnn_type):
    params_dict = {
        'GCN': get_gcn_params,
        'GIN': get_gin_params,
        'GAT': get_gat_params
    }
    return params_dict[gnn_type]()