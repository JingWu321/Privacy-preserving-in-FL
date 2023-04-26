#!/usr/bin/env python
import flwr as fl
from flwr.server.server import Server
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager, ClientManager
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_weights,
)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import timm
from timm.utils import accuracy, AverageMeter
from torch.utils.data import Subset

from collections import OrderedDict
import numpy as np
import argparse
from typing import List, Optional, Tuple, Dict
import random
import sys
import time
import datetime
from pathlib import Path

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger
)
from utils.net import (
    LeNet_MNIST,
    ConvNet
)
from utils import comm
from utils.load_celeba import CelebA
from utils.resnet import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='test under federated learning framework')
    # Dataset
    parser.add_argument('--root', default='/mnt/data/dataset', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--ckp_path', default='./weights/model_round_100.pth', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--method', default='iid', type=str)
    parser.add_argument('--train_lr', default=0.01, type=float)  # 0.001 for cifar10 non-iid
    # parameter for defenses
    parser.add_argument('--rnd', default=1, type=int)
    parser.add_argument('--defense', default='', type=str)
    parser.add_argument('--version', default='', type=str)
    parser.add_argument('--startpoint', default='none', type=str)
    # PRECODE, notice that to compare with precode, shuffle in dataloader need to be False
    parser.add_argument('--precode_size', default=256, type=int)
    # ATS
    parser.add_argument('--aug_list', default='21-13-3+7-4-15', type=str)
    # server
    parser.add_argument('--strategy', default='random', type=str)
    parser.add_argument('--minfit', default=10, type=int)
    parser.add_argument('--mineval', default=1, type=int)
    parser.add_argument('--minavl', default=60, type=int)
    parser.add_argument('--num_rounds', default=100, type=int)
    # clients
    parser.add_argument('--subset', dest='subset', default='random', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=100, type=int)
    parser.add_argument('--DevNum', dest='DevNum', default=5, type=int)
    args = parser.parse_args()
    return args



### ==== MyClientManager (Under Development) ==== ###
class MyClientManager(SimpleClientManager):
    def fit_sample(
        self,
        parameters: Parameters,
        num_clients: int,
        min_num_clients: Optional[int] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        print('num_clients-{}, min_num_clients-{}, clients-{}'.format(
            num_clients, min_num_clients, len(self.clients)
        ))
        # Sample clients which meet the criterion
        available_cids = list(self.clients)

        if len(self.clients) > 1:
            loss_list = []
            for cid in available_cids:
                # params_res = self.clients[cid].get_parameters()
                # eval_ins = EvaluateIns(params_res.parameters, {"num_rounds": 1})
                # evaluate_res = self.clients[cid].evaluate(eval_ins)
                # print('[Client]-{}, [Loss]-{:.4f}'.format(cid, evaluate_res.loss))
                # fit_ins = FitIns(params_res.parameters, {"num_rounds": 1})
                fit_ins = FitIns(parameters, {"num_rounds": 1})
                fit_res = self.clients[cid].fit(fit_ins)
                print('[Client]-{}, [Loss]-{:.4f}'.format(cid, fit_res.metrics['loss']))
                loss_list.append(fit_res.metrics['loss'])
            idx = torch.argsort(torch.Tensor(loss_list))[-30:]
            sampled_cids = [available_cids[idxx] for idxx in idx]
        else:
            sampled_cids = random.sample(available_cids, num_clients)

        print('[sampled_clients for next round]:', sampled_cids)
        return [self.clients[cid] for cid in sampled_cids]
### \\==== MyClientManager (Under Development) ====// ###

class SaveModelAndMetricsStrategy(fl.server.strategy.FedAvg):

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: MyClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.fit_sample(parameters=fit_ins.parameters,
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]], # FitRes is like EvaluateRes and has a metrics key
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple

        if (rnd % 50 == 0) and aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = parameters_to_weights(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), "./weights/model_round_%d.pth" % (rnd))

        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Weigh accuracy of each client by number of examples used
        accuracies = list(map(lambda r: r[1].metrics['accuracy'], results))
        examples = list(map(lambda r: r[1].num_examples, results))
        losses = list(map(lambda r: r[1].loss, results))
        print('Round {:03d} | FL_loss {:.4f} | FL_acc {:.4f} | Test images {}'.format(
              rnd, losses[0], accuracies[0], examples[0]))
        if rnd == 1:
            self.best_acc = 0.
            self.best_rnd = 0
        if self.best_acc < accuracies[0]:
            self.best_acc = accuracies[0]
            self.best_rnd = rnd
        if rnd == Total_rnds:
            print(f'Best FL accuracy: {self.best_acc} on round {self.best_rnd}')

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


class SaveModelAndMetricsStrategy_random(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]], # FitRes is like EvaluateRes and has a metrics key
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple

        if (rnd % 50 == 0) and aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = parameters_to_weights(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), "./weights/model_round_%d.pth" % (rnd))

        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Weigh accuracy of each client by number of examples used
        accuracies = list(map(lambda r: r[1].metrics['accuracy'], results))
        examples = list(map(lambda r: r[1].num_examples, results))
        losses = list(map(lambda r: r[1].loss, results))
        print('Round {:03d} | FL_loss {:.4f} | FL_acc {:.4f} | Test images {}'.format(
              rnd, losses[0], accuracies[0], examples[0]))
        if rnd == st_rnd:
            self.best_acc = 0.
            self.best_rnd = 0
        if self.best_acc < accuracies[0]:
            self.best_acc = accuracies[0]
            self.best_rnd = rnd
        if rnd == Total_rnds:
            print(f'Best FL accuracy: {self.best_acc} on round {self.best_rnd}')

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": 1,
         "lr": 0.01*(0.99**server_round),
    }
    return config


if __name__ == '__main__':
    args = parse_args()
    Path('./server_logs').mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger('server_logs/' + args.dataset + '_' + args.defense + '_sever.csv', sys.stdout)
    print(args)

    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Load Model
    if args.dataset == 'MNIST':
        net = LeNet_MNIST()
    elif args.dataset == 'CIFAR10':
        net = ConvNet(width=32, num_classes=10, num_channels=3)
    elif args.dataset == 'CelebA':
        net = resnet18(pretrained=args.model_pretrained)
        fc = getattr(net, 'fc')
        feature_dim = fc.in_features
        setattr(net,'fc', torch.nn.Linear(feature_dim, 2))
    net.to(**setup)

    Total_rnds = args.num_rounds
    st_rnd = args.rnd
    if args.pretrained:
        # Load Pre-trained Model
        ckp = torch.load(args.ckp_path)
        init_params = [val.cpu().numpy() for _, val in ckp.items()]

        # Create strategy and run server
        strategy = SaveModelAndMetricsStrategy_random(
            fraction_fit=0.1, # Sample 10% of available clients for the next round
            min_fit_clients=args.minfit, # Minimum number of clients to be sampled for the next round
            fraction_eval=0.01, # Fraction of clients used during validation
            min_eval_clients=args.mineval, # Minimum number of clients used during validation
            min_available_clients=args.minavl, # Minimum number of clients that need to be connected to the server before a training round can start
            on_fit_config_fn=fit_config, # Function that returns the training configuration for each round
            initial_parameters=init_params, # Initial model parameters
        )
    else:
        # Create strategy and run server
        strategy = SaveModelAndMetricsStrategy_random(
            fraction_fit=0.1, # Sample 10% of available clients for the next round
            min_fit_clients=args.minfit, # Minimum number of clients to be sampled for the next round
            fraction_eval=0.01, # Fraction of clients used during validation
            min_eval_clients=args.mineval, # Minimum number of clients used during validation
            min_available_clients=args.minavl, # Minimum number of clients that need to be connected to the server before a training round can start
            on_fit_config_fn=fit_config, # Function that returns the training configuration for each round
        )
    start_time = time.time()
    fl.server.start_server("[::]:8080", config={"num_rounds": args.num_rounds}, strategy=strategy)

    # Print final timestamp
    print('Defense Method: ', args.defense)
    print('Strategy Method: ', args.strategy)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
