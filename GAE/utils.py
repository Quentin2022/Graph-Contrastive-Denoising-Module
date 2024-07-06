import numpy as np
import h5py
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def load_data(filepath="processed_data.h5",
              graph_type='G_PCC', if_normalized_adj=False, if_plot_adj=False):
    
    with h5py.File(filepath, 'r') as file:
        node_features = np.transpose(file['node_features'][:], (0, 2, 1))
        labels = file['labels'][:]
        labels = convert_labels_to_int(labels)

        if graph_type in file.keys():
            graph_structures = file[graph_type][:]
            for i in range(len(graph_structures)):
                np.fill_diagonal(graph_structures[i], 0)  # remove self-loop
            if if_normalized_adj:
                graph_structures = normalize_adjacency(graph_structures)
        else:
            raise ValueError(f"Graph type '{graph_type}' not found in the file.")
        
    node_features = preprocess_features(node_features)

    # visualization
    if if_plot_adj:
        visualize_graph_structures(graph_structures, labels)

    datasets = ten_fold_split(node_features, labels, graph_structures)

    return datasets

def visualize_graph_structures(graph_structures, labels, num_samples_per_class=5):
    unique_labels = np.unique(labels)
    fig, axes = plt.subplots(len(unique_labels), num_samples_per_class, figsize=(15, 3 * len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        if len(indices) < num_samples_per_class:
            chosen_indices = np.random.choice(indices, num_samples_per_class, replace=True)
        else:
            chosen_indices = np.random.choice(indices, num_samples_per_class, replace=False)
        
        for j, idx in enumerate(chosen_indices):
            ax = axes[i][j] if len(unique_labels) > 1 else axes[j]
            ax.imshow(graph_structures[idx], cmap='hot', interpolation='nearest')
            ax.set_title(f"Label: {label}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def preprocess_features(features):
    features[np.isnan(features)] = np.median(features[~np.isnan(features)])
    features[np.isinf(features)] = np.max(features[~np.isinf(features)])

    scaler = StandardScaler()
    num_samples, num_channels, feature_dim = features.shape
    features = features.reshape(-1, feature_dim)
    features = scaler.fit_transform(features)
    features = features.reshape(num_samples, num_channels, feature_dim)
    return features

def normalize_adjacency(adj_matrices):
    for i in range(len(adj_matrices)):
        np.fill_diagonal(adj_matrices[i], 0)
        row_sums = adj_matrices[i].sum(axis=1)
        adj_matrices[i] = adj_matrices[i] / row_sums[:, np.newaxis]
        adj_matrices[i][np.isnan(adj_matrices[i])] = 0
    return adj_matrices

def convert_labels_to_int(labels):
    result = []
    for label in labels:
        if label == b'preical':
            result.append(0)
        elif label == b'ical':
            result.append(1)
        else:
            raise ValueError("Label must be b'preical' or b'ical'")
    return np.array(result)

def ten_fold_split(node_features, labels, graphs, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    datasets = []

    for train_idx, test_idx in kf.split(node_features):
        train_features = node_features[train_idx]
        test_features = node_features[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        train_graphs = graphs[train_idx]
        test_graphs = graphs[test_idx]

        datasets.append({
            'train': {
                'features': torch.tensor(train_features, dtype=torch.float32),
                'labels': torch.tensor(train_labels, dtype=torch.long),
                'graphs': torch.tensor(train_graphs, dtype=torch.float32)
            },
            'test': {
                'features': torch.tensor(test_features, dtype=torch.float32),
                'labels': torch.tensor(test_labels, dtype=torch.long),
                'graphs': torch.tensor(test_graphs, dtype=torch.float32)
            }
        })
    return datasets
