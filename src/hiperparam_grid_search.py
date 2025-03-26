from itertools import product

from pathlib import Path

import json

from models.VanillaRNNModel import VanillaRNNModel
from models.LSTMModel import LSTMModel
from models.BaselineModel import BaselineModel

import torch

# create dirs
Path("./output/baseline").mkdir(parents=True, exist_ok=True)
Path("./output/RNN").mkdir(parents=True, exist_ok=True)
Path("./output/LSTM").mkdir(parents=True, exist_ok=True)

# create model param combinations
max_size_list = [-1, 10000, 8000, 6000, 4000]
lr_list = [1e-2, 1e-3, 1e-4]
grad_clip_list = [0.025, 0.25, 0.5, 2.5]
epoch_list = [5, 10, 15]

# "product" function from itertools
combinations = list(product(max_size_list, lr_list, grad_clip_list, epoch_list))

# prepare model classes and model names for ZIP
model_classes = [BaselineModel, VanillaRNNModel, LSTMModel]
model_names = ["baseline", "RNN", "LSTM"]

# loop over model types
for model_class, model_name in zip(model_classes, model_names):
    
    # best * vars init
    best_accuracy = None
    best_model = None
    best_config = None
    
    print(f"Gridsearch for {model_name} model started")

    for combination in combinations:
        
        # create config variable for model init
        config = {
            "max_size": combination[0],
            "lr": combination[1],
            "grad_clip": combination[2],
            "num_epoch": combination[3]
        }
        
        # model class is callable
        model = model_class(config)
        
        # train with verbose=True
        train_info = model.train_(True)
        
        # get latest test dataset accuracy
        test_accuracy = train_info[-1][-1]
        
        # keep best model (by test accuracy)
        if best_model is None or test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_config = config
    
    print(f"Gridsearch for {model_name} model ended. Saving best model...")

    # save best combination
    with open(f"./output/{model_name}/config.cfg", 'w') as f:
        f.write(json.dumps(best_config))

    # save best model's params to file
    torch.save(best_model.state_dict(), f"./output/{model_name}/model.mdl")