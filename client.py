import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import timm
from timm.utils import accuracy, AverageMeter
from collections import OrderedDict
from torch.utils.data import Subset

import argparse
import numpy as np
import sys
import gc
import time
import datetime
import flwr as fl
from pathlib import Path

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger
)
from utils.net import (
    LeNet_MNIST,
    ConvNet,
)
from defense import defenses_para
from utils import comm
from utils.load_celeba import CelebA
from utils.resnet import resnet18


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():

    parser = argparse.ArgumentParser(description='Test using clients')
    # Dataset
    parser.add_argument('--root', default='/mnt/data/dataset', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--normalize', default=True, action='store_false')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--method', default='iid', type=str)
    parser.add_argument('--train_lr', default=0.01, type=float)  # 0.001 for cifar10 non-iid
    parser.add_argument('--cost_fn', default='sim', type=str)
    # clients
    parser.add_argument('--subset', dest='subset', default='random', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=100, type=int)
    parser.add_argument('--DevNum', dest='DevNum', default=5, type=int)
    # parameter for defenses
    parser.add_argument('--rnd', default=1, type=int)
    parser.add_argument('--defmd', default=False, action='store_true')
    parser.add_argument('--defense', default='', type=str)
    # parameter for soteria and compression defense
    parser.add_argument('--percent_num', default=70, type=int, help='1-40 for soteria and 1-80 for compresion.')
    parser.add_argument('--layer_num', default=6, type=int, help='(8) for cifar10, mnist 10 (12 cifar10) for imprintattack with perturb_imprint is False, 1 for True.')
    parser.add_argument('--perturb_imprint', action='store_true')  # default is False
    # parameter for dp defense
    parser.add_argument('--noise_name', default='Gaussian', type=str)
    parser.add_argument('--loc', default=0., type=float)
    parser.add_argument('--scale', default=1e-2, type=float, help='from 1e-4 to 1e-1.')
    # parameter for our defense
    parser.add_argument('--version', default='', type=str)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--num_sen', default=1, type=int, help='<= num_imgs / 2.')
    parser.add_argument('--per_adv', default=1, type=int, help='>= num_sen.')
    parser.add_argument('--min_idx', default=0, type=int, help='0 for out, 1 for fea, 2 for tmp.')
    parser.add_argument('--decost_fn', default='sim', type=str)
    parser.add_argument('--demax_iter', default=1000, type=int)
    parser.add_argument('--delr', default=0.1, type=float)
    parser.add_argument('--delr_decay', default=True, action='store_false')
    parser.add_argument('--deg', default=1., type=float, help='control the adv_g.')
    parser.add_argument('--alpha', default=0.1, type=float, help='control the contribution from x_sim.')
    parser.add_argument('--beta', default=0.001, type=float, help='control the contribution from fx_sim.')
    parser.add_argument('--lamb', default=0.3, type=float, help='label mixup.')
    parser.add_argument('--detv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--deboxed', default=False, action='store_true')
    parser.add_argument('--startpoint', default='none', type=str)
    # PRECODE, notice that to compare with precode, shuffle in dataloader need to be False
    parser.add_argument('--precode_size', default=256, type=int)
    # ATS
    parser.add_argument('--aug_list', default='21-13-3+7-4-15', type=str)
    args = parser.parse_args()
    return args


def load_data(args, dataset=None, batch_size=None):
    if dataset is None:
        dataset = args.dataset
    """Load dataset (training and test set)."""
    if dataset == 'MNIST':
        data_mean = (0.13066047430038452, )
        data_std = (0.30810782313346863,)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = MNIST(root=args.root, train=True, download=True,
                         transform=data_transform)
        testset = MNIST(root=args.root, train=False, download=True,
                        transform=data_transform)
    elif dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR10(root=args.root, train=True, download=True,
                           transform=data_transform)
        testset = CIFAR10(root=args.root, train=False, download=True,
                          transform=data_transform)
        if args.defense == 'ats':
            trainset.transform = comm.build_transform(
                data_mean, data_std, comm.split(args.aug_list))
    elif args.dataset == 'CelebA':
        data_mean = (0.5, 0.5, 0.5)
        data_std = (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        trainset = CelebA(root=args.root + '/CelebA', train=True,
                          transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                            normalize,
                          ]))
        testset = CelebA(root=args.root + '/CelebA', train=False,
                         transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.CenterCrop(32),
                           transforms.ToTensor(),
                           normalize,
                          ]))
    else:
        assert False, 'not support the dataset yet.'

    # split dataset for each client
    train_y = np.array(trainset.targets)
    data_idx = [[] for _ in range(args.TotalDevNum)]
    if args.method == 'iid':
        idxs = np.random.permutation(len(trainset.data))
        data_idx = np.array_split(idxs[:20000], args.TotalDevNum)
    elif args.method == 'non-iid':
        class_idx = [np.where(train_y==i)[0] for i in range(10)]
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(10), 2, replace=False)
            len0 = len(class_idx[idxs[0]])
            len1 = len(class_idx[idxs[1]])
            num = 200
            idxx0 = torch.randint(0, len0, (num,))
            idxx1 = torch.randint(0, len1, (num,))
            data_idx[i] = class_idx[idxs[0]][idxx0].tolist() + class_idx[idxs[1]][idxx1].tolist()
    # For CelebA
    # if args.method == 'iid':
    #     idxs = np.random.permutation(len(trainset.images))
    #     data_idx = np.array_split(idxs[:2000], args.TotalDevNum)
    # elif args.method == 'non-iid':
    #     class_idx = [np.where(train_y==i)[0] for i in range(10177)]
    #     numperclient = 10  # each cleint has <= n identities
    #     for i in range(args.TotalDevNum):
    #         idxs = np.random.choice(range(10177), numperclient, replace=False)
    #         for j in range(numperclient):
    #             len0 = len(class_idx[idxs[j]])
    #             # print(f'ID {idxs[j]} has {len0} images.')
    #             if len0 > 1:
    #                 idxx = torch.randint(0, len0, (len0,))
    #                 data_idx[i] += class_idx[idxs[j]][idxx].tolist()
    #             else:
    #                 continue
    train_subset = Subset(trainset, data_idx[args.DevNum - 1])

    if batch_size is None:
        batch_size = args.batch_size
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    num_examples = {"trainset" : len(train_subset), "testset" : len(testset), "dm": data_mean, "ds": data_std}
    return trainloader, testloader, num_examples

def train(loss_fn, optimizer, model, trainloader, epochs, dm, ds, args, sp_trainloader, rnd):
    """Train the network on the training set."""
    model.train()
    train_losses = AverageMeter()


    for epoch_idx in range(epochs):
        model.train()
        for batch_idx, (gt_imgs, gt_labels) in enumerate(trainloader):
            gt_imgs, gt_labels = gt_imgs.to(device), gt_labels.to(device)
            if args.startpoint == 'noise':
                noise = torch.randn_like(gt_imgs[-args.num_sen - (args.num_sen * args.per_adv):-args.num_sen])
                gt_imgs[-args.num_sen - (args.num_sen * args.per_adv):-args.num_sen] = noise
            elif args.startpoint == 'QMNIST':
                for j, (qimgs, qlabels) in enumerate(sp_trainloader):
                    qimgs = qimgs.to(device)
                    qlabels = qlabels.to(device)
                    if j < batch_idx:
                        continue
                    else:
                        gt_imgs[-args.num_sen - (args.num_sen * args.per_adv):-args.num_sen] = qimgs
                        gt_labels[-args.num_sen - (args.num_sen * args.per_adv):-args.num_sen] = qlabels
                        break

            # perturb
            if args.defmd and (rnd >= args.rnd):
                # commput gt_gradient
                model.eval()
                model.zero_grad()
                if args.defense == 'ours':
                    assert len(gt_imgs) > (args.per_adv * args.num_sen + args.num_sen - 0.1), 'not enough imgs to be modified.'
                    out, _, _ = model(gt_imgs[:-args.num_sen])
                    gt_loss = loss_fn(out, gt_labels[:-args.num_sen])
                else:
                    out, _, _ = model(gt_imgs)
                    gt_loss = loss_fn(out, gt_labels)
                gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
                gt_gradient = [grad.detach().clone() for grad in gt_gradients]
                torch.cuda.empty_cache()
                gc.collect()

                # defense
                if args.defense == 'soteria':
                    gt_gradient = defenses_para.defense_soteria(args,
                        gt_imgs, gt_labels, model, loss_fn, device,
                        layer_num=args.layer_num, percent_num=args.percent_num,
                        perturb_imprint=args.perturb_imprint)
                elif args.defense == 'compression':
                    gt_gradient = defenses_para.defense_compression(
                        gt_gradients, device, percent_num=args.percent_num)
                elif args.defense == 'dp':
                    gt_gradient = defenses_para.defense_dp(
                        gt_gradients, device, loc=args.loc, scale=args.scale, noise_name=args.noise_name)
                elif args.defense == 'ours':  # optimize-based defense for all attacks
                    gt_gradient, adv_imgs, adv_labels = defenses_para.defense_optim(
                        args, model, loss_fn, gt_gradients, gt_imgs, gt_labels, dm, ds, device)
                else:
                    pass
                torch.cuda.empty_cache()
                gc.collect()

                model.train()
                # backward using adv_imgs
                optimizer.zero_grad()
                out, _, _ = model(gt_imgs)
                loss = loss_fn(out, gt_labels)
                loss.backward()

                # overwrite current param
                pointer = 0
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.copy_(gt_gradient[pointer].view_as(p))
                    pointer += 1
            else:
                print('No defenses!')
                optimizer.zero_grad()
                out, _, _ = model(gt_imgs)
                loss = loss_fn(out, gt_labels)
                loss.backward()

            optimizer.step()
            train_losses.update(loss, gt_imgs.size(0))
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    return train_losses.avg.item()

def test(criterion, net, testloader):
    """Validate the network on the entire test set."""
    net.eval()
    top1 = AverageMeter()
    test_losses = AverageMeter()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _, _ = net(images)
            acc1 = timm.utils.accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], images.size(0))
            loss = criterion(outputs, labels)
            test_losses.update(loss, images.size(0))
    test_loss = test_losses.avg.item()
    accuracy = top1.avg.item()
    return test_loss, accuracy

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, net, args):
        super(MNISTClient, self).__init__()
        self.net = net
        self.args = args
        self.trainloader, self.testloader, self.num_examples = load_data(args)
        self.dm = torch.as_tensor(self.num_examples['dm'], **setup)[None, :, None, None]  # 1xCx1x1
        self.ds = torch.as_tensor(self.num_examples['ds'], **setup)[None, :, None, None]
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.train_lr, momentum=0.9)

        # load starpoint
        self.sp_trainloader = None
        if args.startpoint == 'QMNIST':
            self.sp_trainloader, _, _ = load_data(args, dataset='QMNIST', batch_size=args.num_sen * args.per_adv)
        print(f'start with {args.startpoint}')

    def get_parameters(self):
        '''return the model weight as a list of NumPy ndarrays'''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        '''update the local model weights with the parameters received from the server'''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        '''set the local model weights, train the local model,
           receive the updated local model weights'''
        self.set_parameters(parameters)
        self.optimizer.param_groups[0]['lr'] = config['lr']
        loss = train(self.loss_fn, self.optimizer,
                     self.net, self.trainloader, epochs=config['local_epochs'],
                     dm=self.dm,
                     ds=self.ds,
                     args=self.args,
                     sp_trainloader=self.sp_trainloader,
                     rnd=config['current_round'])
        print('Device {:2d} | Train loss {:.4f} | LR {:.4f} | Train images {:} | Round {:3d}'.format(
            self.args.DevNum, loss, self.optimizer.param_groups[0]['lr'], self.num_examples["trainset"], config['current_round']))
        return self.get_parameters(), self.num_examples["trainset"], {"loss": float(loss)}

    def evaluate(self, parameters, config):
        '''test the local model'''
        self.set_parameters(parameters)
        loss, accuracy = test(self.loss_fn, self.net, self.testloader)
        print('Device {:2d} | Test loss {:.4f} | Test accuracy {:.4f} | Test images {:}'.format(
            self.args.DevNum, loss, accuracy, self.num_examples["testset"]))
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

if __name__ == '__main__':

    args = parse_args()
    Path('./client_logs').mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger('client_logs/client' + '_' + str(args.DevNum) + '.csv', sys.stdout)
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

    start_time = time.time()
    fl.client.start_numpy_client("[::]:8080", client=MNISTClient(net, args))

    # Print final timestamp
    print('Defense Method: ', args.defense)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
