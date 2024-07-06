import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import load_data
from models import GCN, GAE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from models_classification import MLP

from pretraining import train_pretraining_model


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

batch_size = 256
num_classes = 2
num_nodes = 16
if_normalized_adj = False
if_plotadj = False
graph_type = 'G_PCC'
if_pretrain = True

pretraining_lr = 0.001
pretraining_epochs = 30
embeddings_layers = [240, 240]
pretraining_weight_decay = 0
embeddings_dropout_rate = 0.5
sim_metric = 'CosSimilarity'
embeddings_loss_type = 'InfoNCELoss'
margin = 0.3

learning_rate = 0.001
epochs_GAE = 30
layer_sizes = [10, 16, 4]
weight_decay = 0.001
dropout = 0.5
readout = 'fc'
TransAdj = 'norL'
embedding_if_finetune = False
finetune_lr = 0.0005
smoothing_rate = 0.1
encode_dim = 128

learning_rate = 0.001
epochs_classification = 30
batch_size = 256
weight_decay = 5e-4
dropout_rate = 0.5
layers = [128, 128]
num_classes = 2

if if_pretrain == False:
    TransAdj = 'A'

datasets = load_data(graph_type=graph_type, if_normalized_adj=if_normalized_adj, if_plot_adj=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(train_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def compute_embeddings_and_labels(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    embeddings = []
    labels_list = []
    with torch.no_grad():  # No need to track gradients
        for features, labels, adj in data_loader:
            features, adj = features.to(device), adj.to(device)
            embeddings.append(model.encoder(features, adj).cpu())  # Compute embeddings and move to CPU
            labels_list.append(labels)

    # Concatenate all embeddings and labels from batches
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels

def evaluate(model_classification, dataloader, device):
    model_classification.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            output = model_classification(features)
            # loss = F.nll_loss(output, labels)
            loss = label_smoothing_loss(output, labels, smoothing=smoothing_rate, classes=num_classes)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss_avg = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return loss_avg, accuracy

def evaluate2(model_classification, dataloader, device):
    model_classification.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            output = model_classification(features)
            # loss = F.nll_loss(output, labels)
            loss = label_smoothing_loss(output, labels, smoothing=smoothing_rate, classes=num_classes)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            probabilities = torch.nn.functional.softmax(output, dim=1)[:, 1]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    loss_avg = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return loss_avg, accuracy, all_labels, all_predictions, all_probabilities


def label_smoothing_loss(outputs, targets, smoothing=0.1, classes=2):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    with torch.no_grad():
        true_dist = torch.zeros_like(outputs)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), confidence)
    return F.kl_div(F.log_softmax(outputs, dim=1), true_dist, reduction='batchmean')

def normalize_adjacency_matrices(adj):
    # max-min normalization
    min_vals = adj.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_vals = adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    
    eps = 1e-5
    ranges = max_vals - min_vals + eps
    normalized_adj = (adj - min_vals) / ranges
    return normalized_adj


fold_accuracies = []
sensitivities = []
specificities = []
auc_scores = []

for fold, dataset in enumerate(datasets):
    print(f"Training fold {fold+1}/{len(datasets)}")
    train_features = dataset['train']['features']
    train_labels = dataset['train']['labels']
    train_graphs = dataset['train']['graphs']

    test_features = dataset['test']['features']
    test_labels = dataset['test']['labels']
    test_graphs = dataset['test']['graphs']
    
    if not if_pretrain:
        train_graphs = normalize_adjacency_matrices(train_graphs)
        test_graphs = normalize_adjacency_matrices(test_graphs)

    train_data_pre = TensorDataset(train_features, train_labels, train_graphs)
    test_data_pre = TensorDataset(test_features, test_labels, test_graphs)
    train_loader_pre, test_loader_pre = create_dataloaders(train_data_pre, test_data_pre, batch_size)

    # Pretraining step
    trained_params = None
    if fold != 0:
        if_plotadj = False

    if if_pretrain:
        print("Starting pretraining...")

        pretrain_losses, pretest_losses, trained_params, trans_train_graphs, trans_test_graphs = train_pretraining_model(
            train_loader_pre, test_loader_pre, num_nodes=num_nodes, epochs=pretraining_epochs, hidden_layers=embeddings_layers,
            dropout_rate=embeddings_dropout_rate, sim_metric=sim_metric, margin=margin, pretraining_lr=pretraining_lr,
            pretraining_weight_decay=pretraining_weight_decay, embeddings_loss_type=embeddings_loss_type, if_plotadj=if_plotadj,
            train_graphs=train_graphs, test_graphs=test_graphs)
        
        # trans_train_graphs = normalize_adjacency_matrices(trans_train_graphs)
        # trans_test_graphs = normalize_adjacency_matrices(trans_test_graphs)
        
    train_data = TensorDataset(train_features, train_labels, train_graphs)
    test_data = TensorDataset(test_features, test_labels, test_graphs)
    train_loader, test_loader = create_dataloaders(train_data, test_data, batch_size)

    if if_pretrain:
        train_data = TensorDataset(train_features, train_labels, trans_train_graphs)
        test_data = TensorDataset(test_features, test_labels, trans_test_graphs)
        train_loader, test_loader = create_dataloaders(train_data, test_data, batch_size)

    # Initialize model
    encoder = GCN(layer_sizes=layer_sizes, dropout=dropout, readout=readout, encode_dim=encode_dim,
                num_nodes=num_nodes, ifTransAdj=False, TransAdj=TransAdj, embedding_hidden_layers=embeddings_layers, 
                embedding_initial_weights=trained_params, embedding_if_finetune=embedding_if_finetune)

    model = GAE(encoder, num_nodes)
    model.to(device)

    optimizer = optim.Adam([
        {'params': [param for name, param in model.named_parameters() if 'graph_embedding.mlp' in name], 'lr': finetune_lr},
        {'params': [param for name, param in model.named_parameters() if 'graph_embedding.mlp' not in name]}
    ], lr=learning_rate, weight_decay=weight_decay)

    # Lists to store performance metrics for plot the training curve
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs_GAE + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for features, labels, adj in train_loader:
            features, labels, adj = features.to(device), labels.to(device), adj.to(device)
            optimizer.zero_grad()

            reconstructed_adj = model(features, adj)
            loss = F.mse_loss(reconstructed_adj, adj)
            
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss / len(train_loader)}")

    embeddings_train, labels_train = compute_embeddings_and_labels(model, train_loader, device)
    embeddings_test, labels_test = compute_embeddings_and_labels(model, test_loader, device)

    train_data_classification = TensorDataset(embeddings_train, labels_train)
    test_data_classification = TensorDataset(embeddings_test, labels_test)
    train_loader_classification, test_loader_classification = create_dataloaders(train_data_classification, test_data_classification, batch_size)

    # Initialize model
    model_classification = MLP(input_dim=encode_dim, layer_sizes=layers, dropout=dropout_rate, num_classes=2)

    model_classification.to(device)
    optimizer = optim.Adam(model_classification.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Lists to store performance metrics for plot the training curve
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs_classification + 1):
        model_classification.train()
        total_loss = 0
        correct = 0
        total = 0
        for features, labels in train_loader_classification:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model_classification(features)
            # loss = F.nll_loss(output, labels)
            loss = label_smoothing_loss(output, labels, smoothing=smoothing_rate, classes=num_classes)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        test_loss, test_accuracy = evaluate(model_classification, test_loader_classification, device)
        
        print(f"Fold {fold+1}, Epoch {epoch}: Train Acc: {train_accuracy:.2f}%, Train Loss: {train_loss:.5f}, Test Acc: {test_accuracy:.2f}%, Test Loss: {test_loss:.5f}")

        if fold == len(datasets) - 1:
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
    
    test_loss, test_accuracy, y_true, y_pred, y_score = evaluate2(model_classification, test_loader_classification, device)
    fold_accuracies.append(test_accuracy)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = (tp / (tp + fn) if (tp + fn) != 0 else 0) * 100
    specificity = (tn / (tn + fp) if (tn + fp) != 0 else 0) * 100
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    auc_score = roc_auc_score(y_true, y_score)
    auc_scores.append(auc_score)

    print(f"Test results in Fold {fold+1}. Accuracy: {test_accuracy:.2f}%, Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%, AUC: {auc_score:.4f}")


    # If this is the last fold, plot the training and test accuracies and losses
    if fold == len(datasets) - 1:
        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim(50, 100)  # Set the y-axis range from 50% to 100%
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

average_accuracy = np.mean(fold_accuracies)
std_deviation = np.std(fold_accuracies)
formatted_accuracies = [f"{accuracy:.2f}%" for accuracy in fold_accuracies]
print(f"\nFold Accuracies: {formatted_accuracies}")
# print(f"Average Accuracy: {average_accuracy:.2f}%")
# print(f"Standard Deviation: {std_deviation:.2f}%")
print(f"Average Accuracy: {average_accuracy:.2f}%, Standard Deviation: {std_deviation:.2f}%")


average_sensitivity = np.mean(sensitivities)
std_sensitivity = np.std(sensitivities)
formatted_sensitivity = [f"{sens:.2f}%" for sens in sensitivities]

average_specificity = np.mean(specificities)
std_specificity = np.std(specificities)
formatted_specificity = [f"{spec:.2f}%" for spec in specificities]

average_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
formatted_auc = [f"{auc:.4f}" for auc in auc_scores]


print(f"\nFold Secsitivities: {formatted_sensitivity}")
print("Average Sensitivity: {:.2f}%, Standard Deviation: {:.2f}%".format(average_sensitivity, std_sensitivity))
print(f"\nFold Specificities: {formatted_specificity}")
print("Average Specificity: {:.2f}%, Standard Deviation: {:.2f}%".format(average_specificity, std_specificity))
print(f"\nFold AUC: {formatted_auc}")
print("Average AUC: {:.4f}, Standard Deviation: {:.4f}\n".format(average_auc, std_auc))