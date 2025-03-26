# Sentiment analysis in PyTorch
Based on lab exercise 3 of FER's Deep Learning course. Goal of the laboratory exercise is to predict sentiment of IMDb reviews using RNNs.

Topics covered: 
* natural language preprocessing
* recursive neural networks
* hiperparameter grid search
* model comparison

## Model description
Three deep learning models are available in folder src/models: baseline (feed-forward) model, vanilla RNN model and LSTM model.

### Baseline model: 
* feed-forward neural network with three linear layers 
* arhitecture: Linear(300, 150), Linear(150, 150), Linear(150, 1)

### Vanilla RNN model: 
* neural network which combines two RNNs and two linear layers
* arhitecture: RNN(300, 150, layers=2), RNN(150, 150, layers=2), Linear(150, 150), Linear(150, 1)

### LSTM model:
* neural network which combines two LSTMs and two linear layers
* arhitecture: LSTM(300, 150, layers=2), LSTM(150, 150, layers=2), Linear(150, 150), Linear(150, 1)

Folder "data" contains IMDb data for train, test and validation datasets stored in CSV files. It also contains pretrained GloVe-6B-300 embedding matrix which is used by implemented models.

Module "utils.py" defines classes and functions which are necessary for models to work including a custom dataset class, custom vocabulary class and pad_collate function.

### NLPDataset:
* class which abstracts a dataset used by our models
* derives from PyTorch Dataset class, implements ```__getitem__``` so it can be passed to a DataLoader

### Vocab:
* class which represents a dataset's vocabulary
* can encode a list of string of tokens (words) into a list of integer representations

### pad_collate_fn:
* function which is passed to a DataLoader along NLPDataset
* prepares data for a NN's forward pass: pads integer vectors and converts data to tensors 

Script "hiperparam_grid_search.py" performs hiperparameter search on every model and stores best configuration and model state in output folder.

Script "model_comparison.py" visualizes every model's performance on train and test datasets.  
