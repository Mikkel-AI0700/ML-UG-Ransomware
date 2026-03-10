import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
import torch
import torch.nn
import torch.optim as optim
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as torch_func

class LoadData (Dataset):
    def __init__ (self, training_x: str, training_y: str):
        self.train_x = pd.read_csv(training_x).to_numpy()
        self.train_y = pd.read_csv(training_y).to_numpy()

    def __len__ (self):
        return len(self.train_x)

    def __getitem__ (self, index):
        train_x_sample = self.train_x[index]
        train_y_sample = self.train_y[index]
        
        return (
            torch.tensor(train_x_sample, dtype=torch.float32),
            torch.tensor(train_y_sample, dtype=torch.float32)
        )

class FeedForwardRansomware (torch.nn.Module):
    def __init__ (self):
        super(FeedForwardRansomware, self).__init__()
        self.module_dict_linear = torch.nn.ModuleDict({
            "linear_layer_input": Linear(in_features=8, out_features=32, device="cuda"),
            "linear_layer_1": Linear(in_features=32, out_features=64, device="cuda"),
            "linear_layer_2": Linear(in_features=64, out_features=128, device="cuda"),
            "linear_layer_3": Linear(in_features=128, out_features=128, device="cuda"),
            "linear_layer_4": Linear(in_features=128, out_features=128, device="cuda"),
            "linear_layer_5": Linear(in_features=128, out_features=128, device="cuda"),
            "linear_layer_6": Linear(in_features=128, out_features=32, device="cuda"),
            "linear_layer_7": Linear(in_features=32, out_features=32, device="cuda"),
            "linear_layer_output": Linear(in_features=32, out_features=3, device="cuda")
        })

    def forward (self, dataset_samples):
        tensor_output = dataset_samples
        for layer_name, layer_ref in self.module_dict_linear.items():
            tensor_output = layer_ref(tensor_output)
            tensor_output = torch_func.relu(tensor_output)
            
            if "input" in layer_name or "output" in layer_name:
                tensor_output = torch_func.dropout(tensor_output, p=0.15)
            else:
                tensor_output = torch_func.dropout(tensor_output, p=0.50)
                
        return tensor_output
    
def main ():
    train_set = LoadData("../../datasets/tr_x.csv", "../../datasets/tr_y.csv")
    train_load_generator = DataLoader(
        train_set, 
        batch_size=2048, 
        shuffle=True, 
        num_workers=120, 
        pin_memory=True,
        prefetch_factor=50,
        persistent_workers=True
    )

    test_data_x_tensor = torch.from_numpy(
        pd.read_csv("../../datasets/ts_x.csv").to_numpy(), 
    )
    test_data_y = pd.read_csv("../../datasets/ts_y.csv").to_numpy()

    ffr_model = FeedForwardRansomware()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    adam_optim = optim.Adam(ffr_model.parameters(), lr=0.0001)

    ffr_model.to("cuda")

    for epoch in range(650):
        accumulated_criterion_loss = 0.0
        for train_feature, train_label in train_load_generator:
            ffr_model.train()

            train_feature = train_feature.to("cuda")
            train_label = train_label.to("cuda")

            adam_optim.zero_grad()
            model_prediction = ffr_model(train_feature)
            loss = cross_entropy_loss(model_prediction, train_label.flatten().long())
            loss.backward()
            adam_optim.step()

            accumulated_criterion_loss += loss.item()

        averaged_accumulated_loss = accumulated_criterion_loss / len(train_load_generator)
        print(f"Epoch: {epoch} | Averaged loss: {averaged_accumulated_loss:.2f}")

    with torch.no_grad():
        ffr_model.eval()
        test_data_x_tensor = test_data_x_tensor.to(device="cuda", dtype=torch.float32)

        inferenced_predictions = ffr_model(test_data_x_tensor)
        inferenced_predictions = inferenced_predictions.detach().cpu().numpy()
        inferenced_predictions = np.argmax(inferenced_predictions, axis=1)

        model_precision_score = precision_score(test_data_y, inferenced_predictions, average="weighted", zero_division=0)
        print(f"Scikit-Learn precision: {model_precision_score:.2f}")

    torch.save(ffr_model.state_dict(), "training_results/ffr_model_weights.pth")
    print(f"Successfully saved PyTorch model weights")


main()
