import flwr as fl
import numpy as np 
import torch
from collections import OrderedDict
from typing import List, Tuple
from model import train_classifier, test_classifier

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, adi, encoder_tuple, opt):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.adi = adi
        self.encoder_tuple = encoder_tuple
        self.opt = opt

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_classifier(self.net, self.trainloader, self.encoder_tuple, self.opt, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test_classifier(self.net, self.valloader, self.encoder_tuple, self.opt)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


