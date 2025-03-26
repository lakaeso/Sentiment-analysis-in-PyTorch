from abc import ABC
from abc import abstractmethod

import torch

class AbstractModel(ABC):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def train_():
        pass
    
    @abstractmethod
    def forward():
        pass
    
    @abstractmethod
    def predict():
        pass
    
    @abstractmethod
    def get_dataset_accuracy():
        pass
    
    @abstractmethod
    def get_parameters():
        pass