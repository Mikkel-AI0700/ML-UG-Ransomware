import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset
import torch.nn.functional as torch_func

class LoadData (Dataset):
    def __init__ (self, training_x: str, training_y: str):
        self.train_x = pd.read_csv(training_x)
        self.train_y = pd.read_csv(training_y)

    def __len_ (self):
        return len(self.train_x)

    def __getitem__ (self, index):
        train_x_sample = self.train_x.iloc[index]
        train_y_sample = self.train_y.iloc[index]
        
        return (
            torch.tensor(train_x_sample, dtype=torch.float32),
            torch.tensor(train_y_sample, dtype=torch.float32)
        )

class FeedForwardRansomware (torch.nn.Module):
    def __init__ (self):
        super(FeedForwardRansomware, self).__init__()
        self.module_dict_linear = torch.nn.ModuleDict({
            "linear_layer_input": torch.nn.Linear(in_features=14, out_features=32, device="cuda"),
            "linear_layer_1": torch.nn.Linear(in_features=32, out_features=64, device="cuda"),
            "linear_layer_2": torch.nn.Linear(in_features=64, out_features=64, device="cuda"),
            "linear_layer_3": torch.nn.Linear(in_features=64, out_features=64, device="cuda"),
            "linear_layer_4": torch.nn.Linear(in_features=64, out_features=3, device="cuda")
        })

    def forward (self, dataset_samples):
        tensor_output = None
        for layer_name, layer_ref in self.module_dict_linear.items():
            tensor_output = layer_ref(dataset_samples)
            tensor_output = torch.nn.functional.relu(tensor_output)
            
            if "layer" in layer_name:
                tensor_output = torch.nn.functional.dropout(tensor_output, p=0.15)
            else:
                tensor_output = torch.nn.functional.dropout(tensor_output, p=0.50)
                
        return tensor_output
    
def _train_feedforward (
    optuna_trial,
    train_x: str,
    train_y: str,
):
    pass

def main ():
    tr_x = pd.read_csv("../../datasets/tr_x.csv")
    tr_y = pd.read_csv("../../datasets/tr_y.csv")
    ts_x = pd.read_csv("../../datasets/ts_x.csv")
    ts_y = pd.read_csv("../../datasets/ts_y.csv")
    
    

