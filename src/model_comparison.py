import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from models.VanillaRNNModel import VanillaRNNModel
from models.LSTMModel import LSTMModel
from models.BaselineModel import BaselineModel

config = {
    "max_size": 1000,
    "lr": 1e-4,
    "grad_clip": 0.25,
    "num_epoch": 2
}

model_classes = [BaselineModel, VanillaRNNModel, LSTMModel]

fig, ax = plt.subplots(1, 3)

for i, model_class in enumerate(model_classes):
    
    model = model_class(config)
    train_info = model.train_(True)

    epochs = np.array(list([i[0] for i in train_info]))
    train_acc = np.array(list([i[1] for i in train_info]))
    test_acc = np.array(list([i[3] for i in train_info]))
    
    ax[i].plot(epochs, train_acc, color="red", label="train")
    ax[i].plot(epochs, test_acc, color="blue", label="test")
    ax[i].legend(loc="upper left")
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Percentage')
    ax[i].set_ylim(0, 100)

plt.show()