import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import load_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

from pretraining import GCD
from hyperparameters import *
from models_GCN import GCN
from models_GIN import GIN
from models_GAT import GAT


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

args = parse_args_GCD()


########## Basic settings ##########
GNN_type = 'GCN'
graph_type = 'G_PCC'
if_pretrain = True
####################################


batch_size = 256
num_classes = 2
num_nodes = 16
smoothing_rate = 0.1
TransAdj = 'norL'  # Whether to transform the Graph into a Laplace matrix

# load dataset
datasets = load_data(graph_type=graph_type, if_normalized_adj=False, if_plot_adj=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(train_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels, adj in dataloader:
            features, labels, adj = features.to(device), labels.to(device), adj.to(device)
            output = model(features, adj)
            loss = label_smoothing_loss(output, labels, smoothing=smoothing_rate, classes=num_classes)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss_avg = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return loss_avg, accuracy

def evaluate2(model, dataloader, device): # Used to calculate overall accuracy, sensitivity, specificity, AUC, 0 no incidence, 1 incidence
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for features, labels, adj in dataloader:
            features, labels, adj = features.to(device), labels.to(device), adj.to(device)
            output = model(features, adj)
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


    train_data = TensorDataset(train_features, train_labels, train_graphs)
    test_data = TensorDataset(test_features, test_labels, test_graphs)
    train_loader, test_loader = create_dataloaders(train_data, test_data, batch_size)


    # Pretraining step
    trained_params = None
    if fold != 0:
        if_plotadj = False

    model_params = get_model_params(GNN_type)

    ################################################################################################
    if if_pretrain:
        print("Starting pretraining...")

        pretrain_losses, pretest_losses, trained_params = GCD(
            train_loader, test_loader, num_nodes=num_nodes, epochs=args.pretraining_epochs,
            hidden_layers=args.embeddings_layers, dropout_rate=args.embeddings_dropout_rate,
            sim_metric=args.sim_metric, margin=args.margin, pretraining_lr=args.pretraining_lr,
            pretraining_weight_decay=args.pretraining_weight_decay, gamma=model_params['gamma'],
            embeddings_loss_type=args.embeddings_loss_type, if_plotadj=args.if_plotadj)
    ################################################################################################
    # The variable trained_params is the parameter obtained from the pre-training

    # Initialize model
    if GNN_type == 'GCN':
        model = GCN(
            layer_sizes=model_params['layer_sizes'], dropout=model_params['dropout'], readout=model_params['readout'], 
            num_classes=num_classes, num_nodes=num_nodes, ifTransAdj=if_pretrain, TransAdj=TransAdj, 
            embedding_hidden_layers=args.embeddings_layers, embedding_initial_weights=trained_params, 
            embedding_if_finetune=model_params['embedding_if_finetune']
        )
    elif GNN_type == 'GIN':
        model = GIN(
            layer_sizes=model_params['layer_sizes'], dropout=model_params['dropout'], readout=model_params['readout'], 
            num_classes=num_classes, num_nodes=num_nodes, ifTransAdj=if_pretrain, TransAdj=TransAdj, 
            embedding_hidden_layers=args.embeddings_layers, embedding_initial_weights=trained_params, 
            embedding_if_finetune=model_params['embedding_if_finetune']
        )
    elif GNN_type == 'GAT':
        model = GAT(
            layer_sizes=model_params['layer_sizes'], dropout=model_params['dropout'], readout=model_params['readout'], 
            num_classes=num_classes, num_nodes=num_nodes, num_heads=model_params['heads'], ifTransAdj=if_pretrain, TransAdj='A', 
            embedding_hidden_layers=args.embeddings_layers, embedding_initial_weights=trained_params, 
            embedding_if_finetune=model_params['embedding_if_finetune'], neighbor_rate=model_params['Neighbor_rate']
        )

    model.to(device)

    # Resetting the fine-tuning learning rate of the embeddings layer
    optimizer = optim.Adam([
        {'params': [param for name, param in model.named_parameters() if 'graph_embedding.mlp' in name], 'lr': model_params['finetune_lr']},
        {'params': [param for name, param in model.named_parameters() if 'graph_embedding.mlp' not in name]}
    ], lr=model_params['learning_rate'], weight_decay=model_params['weight_decay'])

    # Lists to store performance metrics for plot the training curve
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, model_params['epochs'] + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for features, labels, adj in train_loader:
            features, labels, adj = features.to(device), labels.to(device), adj.to(device)
            optimizer.zero_grad()
            output = model(features, adj)
            loss = label_smoothing_loss(output, labels, smoothing=smoothing_rate, classes=num_classes)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        
        print(f"Fold {fold+1}, Epoch {epoch}: Train Acc: {train_accuracy:.2f}%, Train Loss: {train_loss:.5f}, Test Acc: {test_accuracy:.2f}%, Test Loss: {test_loss:.5f}")

        if fold == len(datasets) - 1:
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
    
    test_loss, test_accuracy, y_true, y_pred, y_score = evaluate2(model, test_loader, device)
    fold_accuracies.append(test_accuracy)

    # Compute confusion matrix
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


# Output the mean and standard deviation of the overall indicator after all folds have been completed
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