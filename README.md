### Graph Contrastive Denoising

Graph Contrastive Denoising (GCD) is a universal GNN module proposed in the paper "GCD: Graph Contrastive Denoising Module for GNNs in EEG Classification." This paper introduces a generic GSL module specifically designed for small-scale, dense, weighted graphs. GCD employs a pretraining and fine-tuning architecture for graph structure learning. The paper has now been published in `Expert System with Application` [（https://linkinghub.elsevier.com/retrieve/pii/S095741742402880X）](https://linkinghub.elsevier.com/retrieve/pii/S095741742402880X).

#### Pretraining Stage
During the pretraining stage, GCD learns a mapping from the original graph to an embedded graph. Local and global features of the embedded graph are extracted, and contrastive learning is applied to bring features of the same class closer together while differentiating features from different classes. Through iterative refinement, the resulting embedded graph retains more relevant information for downstream classification tasks and removes irrelevant noise. The pretrained denoised graph structure becomes more stable, making it better suited for training GNNs.

#### Fine-Tuning Stage
In the fine-tuning stage, the embedding module of GCD is integrated with subsequent GNNs. This stage involves the simultaneous learning of GNN parameters and fine-tuning of GCD parameters for specific downstream GNN tasks.

#### Experimental Results
Experiments on the CHB-MIT public epilepsy dataset show that integrating the GCD module into five types of GNNs—Graph Convolutional Networks, Graph Isomorphism Networks, Local Graph Attention Networks, Global Graph Attention Networks, and Graph Autoencoders—resulted in average accuracy improvements of 9.20%, 7.20%, 6.63%, 5.05%, and 7.39%, respectively.

The experiment uses the CHB-MIT public epilepsy EEG dataset, available at [this link](https://physionet.org/content/chbmit/1.0.0/). The data segments used in the experiments were taken from the pre- and post-seizure periods of this dataset. After preprocessing, node feature extraction, and connectivity construction, the processed data is saved in the file "processed_data.h5".

#### File Descriptions
- `hyperparameters.py`: Sets various model hyperparameters.
- `models_GAT.py`: Defines the GAT model, including its integration with GCD.
- `models_GCN.py`: Defines the GCN model, including its integration with GCD.
- `models_GIN.py`: Defines the GIN model, including its integration with GCD.
- `pretraining.py`: Defines the pretraining module for GCD.
- `train.py`: Guides the training process.
- `utils.py`: Imports and processes data.
- `GAE` folder: Contains the implementation of GAE processes, as GAE follows a different workflow compared to other models.

#### Running the Experiment
1. Set the correct dataset path in `utils.py` to import the "processed_data.h5" file.
2. Use `train.py` to train the model. Key settings include:
   - `GNN_type`: Specifies the type of GNN to use (excluding GAE).
   - `graph_type`: Specifies the graph structure to use. Available options are: 'G_PCC' for Pearson Correlation Coefficient, 'G_MSC' for Magnitude-Squared Coherence, and 'G_PLV' for Phase-Locking Value.
   - `if_pretrain`: Indicates whether to include the GCD module. If set to True, both the pretraining and fine-tuning stages will involve GCD.
  

#### The experiment data "processed_data.h5" can be download from https://drive.google.com/file/d/1SoGdJ7iSehJwy7T9QMhzwCABOfWlBHvh/view?usp=sharing.
