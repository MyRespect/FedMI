import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import numpy as np 
import flwr as fl
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from flwr.common import Metrics
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from model import Classifier
from dataset import load_datasets, parse_option
from client import FlowerClient
from client import set_parameters, get_parameters
from model import train_classifier, test_classifier

num_clients = 5
width, height = 100, 3
batch_size = 32
train_test_ratio = 0.6
classifier_epochs = 30 # server_evaluate
label_num = 5
classifier_train_mode = True
adi = True # set up data imputation
opt = parse_option()
trainloaders, valloaders, testloader, server_trainloader = load_datasets(opt, num_clients, batch_size, train_test_ratio)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # save GPU -.-


input_size = 1024  # multimodal feature embedding
hidden1_size = 512  # hidden layer size
hidden2_size = 128  # Second hidden layer size
num_classes = 11  # Number of output classes: 11 behavior biomarkers

opt.device = device
folder_path = './trial_1/'

audio_encoder = torch.load(folder_path+'audio_encoder.pt', map_location= device)
depth_encoder = torch.load(folder_path+'depth_encoder.pt', map_location= device)
radar_encoder = torch.load(folder_path+'radar_encoder.pt', map_location= device)
autoencoder = torch.load(folder_path+'autoencoder.pt', map_location= device)
contrastive_model = torch.load(folder_path+'contrastive_model.pt', map_location= device)
contrastive_model = 0

encoder_tuple = (audio_encoder, depth_encoder, radar_encoder, autoencoder, contrastive_model)

# The `evaluate` function will be by Flower called after every round
def server_evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    classifier = Classifier().to(device)
    
    # valloader = valloaders[0]

    set_parameters(classifier, parameters)  # Update model with the latest parameters

    loss, accuracy = test_classifier(classifier, testloader, encoder_tuple, opt)

    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    net = Classifier().to(device)

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, adi, encoder_tuple, opt)

def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 5, # here not used in model
    }
    return config

if __name__ == "__main__":

    params = get_parameters(Classifier())

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,  # Sample 100% of available clients for training
        fraction_evaluate=1,  # Sample 100% of available clients for evaluation
        min_fit_clients=4,  # Never sample less than 4 clients for training
        min_evaluate_clients=4,  # Never sample less than 4 clients for evaluation
        min_available_clients=num_clients,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        evaluate_fn=server_evaluate,  # Pass the evaluation function
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if device.type == "cuda":
        client_resources = {"num_gpus": 4}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
        client_resources=client_resources,
    )