import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

import os
import sys
import time
import datetime
import argparse
from pathlib import Path
import numpy as np

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger)
from utils.net import (
    LeNet_MNIST, LeNet_MNIST_imp,
    ConvNet, ConvNet_imp)
from utils.metric import psnr, ssim_batch, lpips_loss
from defense import defenses_para
from attack import attacks


def get_args_parser():
    parser = argparse.ArgumentParser(description='Test attacks and defenses.')
    # Dataset
    parser.add_argument('--root', default='/mnt/data/dataset', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_imgs', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--batch_idx', default=0, type=int)
    parser.add_argument('--vis', default=True, action='store_false')
    # parameter for DLG and GS attack
    parser.add_argument('--attack', default='dlg', type=str)
    parser.add_argument('--imprint', default='no_sparse', type=str)
    parser.add_argument('--cost_fn', default='sim', type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_decay', default=True, action='store_false')
    parser.add_argument('--tv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--boxed', action='store_true')
    # parameter for imprint attack
    parser.add_argument('--bins', default=10, type=int)
    # parameter for defenses
    parser.add_argument('--defense', default='', type=str)
    # parameter for soteria and compression defense
    parser.add_argument('--percent_num', default=70, type=float)
    parser.add_argument('--layer_num', default=6, type=int)
    parser.add_argument('--perturb_imprint', action='store_true')
    # parameter for dp defense
    parser.add_argument('--noise_name', default='Gaussian', type=str)
    parser.add_argument('--loc', default=0., type=float)
    parser.add_argument('--scale', default=1e-2, type=float)
    # parameter for our defense
    parser.add_argument('--projection', action='store_true')
    parser.add_argument('--num_sen', default=1, type=int, help='<= num_imgs / 2.')
    parser.add_argument('--per_adv', default=1, type=int, help='>= num_sen.')
    parser.add_argument('--demax_iter', default=1000, type=int)
    parser.add_argument('--delr', default=0.1, type=float)
    parser.add_argument('--delr_decay', default=True, action='store_false')
    parser.add_argument('--deg', default=1., type=float, help='control the adv_g.')
    parser.add_argument('--alpha', default=0.1, type=float,
    help='control the contribution from x_sim, 1 (30 on MNIST) for imprint attack.')
    parser.add_argument('--beta', default=0.001, type=float,
    help='control the contribution from fx_sim, 10 (100 on MNIST and CIFAR10) for imprint attack.')
    parser.add_argument('--lamb', default=0.3, type=float)
    parser.add_argument('--detv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--deboxed', default=False, action='store_true')
    # clients
    parser.add_argument('--subset', dest='subset', default='random', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=100, type=int)
    parser.add_argument('--DevNum', dest='DevNum', default=5, type=int)
    args = parser.parse_args()
    return args


def load_data(args, dataset=None, batch_size=None):
    """Load dataset (training and test set)."""
    if dataset is None:
        dataset = args.dataset
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
    else:
        assert False, 'not support the dataset yet.'

    # split dataset for each client
    if args.subset == 'random':
        propotion = np.random.dirichlet(np.ones(args.TotalDevNum), size=2)
        split_sets = []
        for pp in propotion[0]:
            split_sets.append(int(pp * len(trainset)))
        split_sets[-1] = len(trainset) - sum(split_sets[:-1])
    else: # equally
        split_sets = (len(trainset) * (np.ones(args.TotalDevNum) / args.TotalDevNum)).astype(int)
    train_subsets = torch.utils.data.random_split(trainset, split_sets,
                        generator=torch.Generator().manual_seed(args.seed))
    train_subset = train_subsets[args.DevNum - 1]
    if batch_size is None:
        batch_size = args.batch_size
    # To compare with defence PRECODE, shuffle=False
    trainloader = DataLoader(train_subset, batch_size=batch_size,
                             shuffle=True, drop_last=True,
                             num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=args.num_workers)
    num_examples = {"trainset" : len(train_subset), "testset" : len(testset), "dm": data_mean, "ds": data_std}
    return trainloader, testloader, num_examples


def load_model(args, setup):
    attacker = None
    server_payload = None
    secrets = None
    loss_fn = nn.CrossEntropyLoss()
    if args.dataset == 'MNIST':
        if args.attack == 'imprint':
            model = LeNet_MNIST_imp()
            if args.trained > 0:
                model, attacker, server_payload, secrets = attacks.Imprint_setting(
                    args, model, loss_fn, setup)
        else:
            model = LeNet_MNIST()
    elif args.dataset == 'CIFAR10':
        if args.attack == 'imprint':
            model = ConvNet_imp(width=32, num_classes=10, num_channels=3)
            if args.trained > 0:
                model, attacker, server_payload, secrets = attacks.Imprint_setting(
                    args, model, loss_fn, setup)
        else:
            model = ConvNet(width=32, num_classes=10, num_channels=3)
    model.to(**setup)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=0.01,
                                      momentum=0.9)
    return loss_fn, model, optimizer_model, attacker, server_payload, secrets


def main(args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device, attacker, server_payload, secrets):

    model.eval()
    model.zero_grad()
    # defense
    if args.defense == 'soteria':
        gt_gradient = defenses_para.defense_soteria(args,
            gt_imgs, gt_labels, model, loss_fn, device,
            layer_num=args.layer_num, percent_num=args.percent_num,
            perturb_imprint=args.perturb_imprint)
    elif args.defense == 'compression':
        gt_gradient = defenses_para.defense_compression(
            gt_imgs, gt_labels, model, loss_fn, device, percent_num=args.percent_num)
    elif args.defense == 'dp':
        gt_gradient = defenses_para.defense_dp(
            gt_imgs, gt_labels, model, loss_fn, device, loc=args.loc, scale=args.scale, noise_name=args.noise_name)
    elif args.defense == 'dcs':
        gt_gradient, adv_imgs, adv_labels = defenses_para.defense_optim(
            args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device)
    else:
        print('No defenses!')
        out, _, _ = model(gt_imgs)
        gt_loss = loss_fn(out, gt_labels)
        gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
        gt_gradient = [grad.detach().clone() for grad in gt_gradients]

    # attack
    if args.attack == 'dlg':
        reconstructed_data = attacks.DLG_attack(
            args, gt_gradient, gt_imgs, gt_labels, model, loss_fn, dm, ds, device)
    elif args.attack == 'imprint':
        reconstructed_data = attacks.Robbing_attack(
            gt_gradient, gt_labels, attacker, server_payload, secrets)
    else:
        print('No attack.')

    # compute metric
    if reconstructed_data is not None:
        st, ed = 3, 3
        output_denormalized = torch.clamp(reconstructed_data * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(gt_imgs * ds + dm, 0, 1)
        test_psnr = psnr(output_denormalized[-args.num_sen:], gt_denormalized[-args.num_sen:], batched=False, factor=1.)
        test_ssim = ssim_batch(output_denormalized[-args.num_sen:], gt_denormalized[-args.num_sen:])
        if args.dataset == 'ImageNet' or args.dataset == 'CelebA':
            test_lpips = lpips_loss(output_denormalized[-args.num_sen:].cpu(), gt_denormalized[-args.num_sen:].cpu())
        else:
            test_lpips = torch.tensor(-0.).to(device)
        print('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}'.format(test_psnr.item(), test_ssim[0].item(), test_lpips.item()))

    return test_psnr, test_ssim, test_lpips


if __name__ == '__main__':
    args = get_args_parser()
    sys.stdout = Logger(args.output_dir  + '/' + args.dataset + '_' + args.attack + '_' + args.defense + '_' + '.csv' , sys.stdout)
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Load Dataset
    trainloader, testloader, num_examples = load_data(args)
    mean, std = num_examples["dm"], num_examples["ds"]
    dm = torch.as_tensor(mean, **setup)[None, :, None, None]  # 1xCx1x1
    ds = torch.as_tensor(std, **setup)[None, :, None, None]
    print('Total images {:d} on {}'.format(num_examples['trainset'], args.dataset))

    # Load model
    loss_fn, model, optimizer_model, attacker, server_payload, secrets = load_model(args, setup)

    mpsnr = []
    mssim = []
    mlpips = []
    start_time = time.time()
    for i, (imgs, labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        test_psnr, test_ssim, test_lpips = main(
            args, model, loss_fn, imgs, labels, dm, ds, device, attacker, server_payload, secrets)

        mpsnr.append(test_psnr)
        mssim.append(test_ssim[0])
        mlpips.append(test_lpips)

    print(torch.mean(torch.stack(mpsnr), dim=0).item(),
          torch.mean(torch.stack(mssim), dim=0).item(),
          torch.mean(torch.stack(mlpips), dim=0).item())


    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
