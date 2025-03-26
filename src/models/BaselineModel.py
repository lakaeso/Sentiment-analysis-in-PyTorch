import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import DataLoader

from utils.utils import NLPDataset, pad_collate_fn 

from .AbstractModel import AbstractModel

class BaselineModel(nn.Module, AbstractModel):
    """Baseline NLP model. Uses no RNNs."""
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # load data
        self.train_dataset = NLPDataset('./data/sst_train_raw.csv', config["max_size"])
        self.valid_dataset = NLPDataset('./data/sst_valid_raw.csv', text_vocab=self.train_dataset.text_vocab)
        self.test_dataset = NLPDataset('./data/sst_test_raw.csv', text_vocab=self.train_dataset.text_vocab)
        
        # network
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)
        
        # params
        self.lr = config["lr"]
        self.grad_clip = config["grad_clip"]
        self.num_epoch = config["num_epoch"]
        self.batch_size_train = 10
        self.batch_size_valid = 10
        self.batch_size_test = 32
        
        # embedding
        self.embedding = self.train_dataset.embedding
        
        # criterion and optim
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.get_parameters(), lr=self.lr)
    
    def train_(self, verbose = False):
        """Starts a training loop."""
        
        train_info = []
        
        for i_epoch in range(1, self.num_epoch + 1):
            loader = DataLoader(self.train_dataset, self.batch_size_train, shuffle=True, collate_fn=pad_collate_fn)
            
            if verbose:
                print(f"... Baseline model starting epoch {i_epoch}")
            
            self.train()
            for batch_num, (x, y_true, length) in enumerate(loader):
                self.zero_grad()
                logits = self.forward(x).flatten()
                loss = self.criterion(logits, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.grad_clip)
                self.optim.step()
            
            train_acc = self.get_dataset_accuracy(self.train_dataset)
            eval_acc = self.get_dataset_accuracy(self.valid_dataset)
            test_acc = self.get_dataset_accuracy(self.test_dataset)

            train_info.append((i_epoch, train_acc, eval_acc, test_acc))
            
            if verbose:
                print(f"... Baseline model epoch {i_epoch}, eval acc={eval_acc:.2f}")
        
        if verbose:
            print(f"... Baseline model finished training, test acc={test_acc:.2f}")
        
        return train_info
                 
    def forward(self, x):
        """Performs a forward pass on input vector x."""
        
        h = x
        h = self.embedding(h)
        h = torch.mean(h, dim=1)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h
    
    def predict(self, x):
        """Gets class predictions for input vector x."""
        
        with torch.no_grad():
            h = self.forward(x)
            h = torch.sigmoid(h)
            h = h.round()
            h = h.flatten()

        return h
    
    def get_dataset_accuracy(self, dataset):
        """Returns accuracy on test dataset."""
        
        self.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, self.batch_size_test, shuffle=True, collate_fn=pad_collate_fn)
            
            no_correct = 0
            no_total = 0
            for batch_num, (x, y_true, length) in enumerate(loader):
                y_pred = self.predict(x)
                
                no_correct += (y_pred == y_true).sum()
                no_total += len(x)
                
            accuracy = no_correct/no_total*100
        
        return accuracy
            
    def get_parameters(self):
        """Gets trainable parameters for this NN."""
        
        p = []
        p.extend(list(self.fc1.parameters()))
        p.extend(list(self.fc2.parameters()))
        p.extend(list(self.fc3.parameters()))
        p.extend(list(self.embedding.parameters()))
        
        return p
                


if __name__ == '__main__':
    b = BaselineModel()
    b.train_(True)