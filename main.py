from cgi import test
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
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
from utils.resnet import resnet18, resnet18_imp
from defense import defenses
from attack import attacks


def get_args_parser():
    parser = argparse.ArgumentParser(description='Test attacks and defenses.')
    # Dataset
    parser.add_argument('--root', default='/mnt/data/dataset', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--normalize', default=True, action='store_false')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_imgs', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--trained', default=-1, type=int)
    parser.add_argument('--train_lr', default=0.01, type=float)
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
    parser.add_argument('--percent_num', default=10, type=float, help='1-40 for soteria and 1-80 for compresion.')
    parser.add_argument('--layer_num', default=6, type=int, help='32 for cifar10, mnist 10 (36 cifar10) for imprintattack with perturb_imprint is False, 1 for True.')
    parser.add_argument('--perturb_imprint', action='store_true')  # default is False
    # parameter for dp defense
    parser.add_argument('--noise_name', default='Gaussian', type=str)
    parser.add_argument('--loc', default=0., type=float)
    parser.add_argument('--scale', default=1e-4, type=float, help='from 1e-4 to 1e-1.')
    # parameter for our defense
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--num_sen', default=1, type=int, help='<= num_imgs / 2.')
    parser.add_argument('--per_adv', default=1, type=int, help='>= num_sen.')
    parser.add_argument('--min_idx', default=0, type=int, help='0 for out, 1 for fea, 2 for tmp.')
    parser.add_argument('--decost_fn', default='sim', type=str)
    parser.add_argument('--demax_iter', default=1000, type=int)
    parser.add_argument('--delr', default=0.1, type=float)
    parser.add_argument('--delr_decay', default=True, action='store_false')
    parser.add_argument('--deg', default=1., type=float, help='control the adv_g.')
    parser.add_argument('--alpha', default=0.1, type=float,
    help='control the contribution from x_sim, 1 (30 on MNIST) for imprint attack.')
    parser.add_argument('--beta', default=0.001, type=float,
    help='control the contribution from fx_sim, 10 (100 on MNIST and CIFAR10) for imprint attack.')
    parser.add_argument('--lamb', default=0.3, type=float, help='label mixup.')
    parser.add_argument('--detv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--deboxed', default=False, action='store_true')
    # clients
    parser.add_argument('--subset', dest='subset', default='random', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=100, type=int)
    parser.add_argument('--DevNum', dest='DevNum', default=5, type=int)
    args = parser.parse_args()
    return args

def load_data(args):
    """Load dataset (training and test set)."""
    if args.dataset == 'MNIST':
        trainset = MNIST(root=args.root, train=True, download=True, transform=transforms.ToTensor())
        testset = MNIST(root=args.root, train=False, download=True, transform=transforms.ToTensor())
        data_mean = (0.13066047430038452, )
        data_std = (0.30810782313346863,)
        if args.normalize is False:
            data_mean = (0.,)
            data_std = (1.,)
    elif args.dataset == 'CIFAR10':
        trainset = CIFAR10(root=args.root, train=True, download=True, transform=transforms.ToTensor())
        testset = CIFAR10(root=args.root, train=False, download=True, transform=transforms.ToTensor())
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        if args.normalize is False:
            data_mean = (0., 0., 0.)
            data_std = (1., 1., 1.)
    elif args.dataset == 'ImageNet':
        data_mean = (0.485, 0.456, 0.406)
        data_std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trainset = ImageFolder(root=args.root + '/' + 'ImageNet/val',
                               transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize,]))
        testset = ImageFolder(root=args.root + '/' + 'ImageNet/val',
                              transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize,]))
    else:
        assert False, 'not support the dataset yet.'

    if args.dataset == 'MNIST' or args.dataset == 'CIFAR10':
        # Organize preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if args.normalize else transforms.Lambda(lambda x: x)])
        trainset.transform = transform
        testset.transform = transform

    # split dataset for each client
    if args.subset == 'random':
        propotion = np.random.dirichlet(np.ones(args.TotalDevNum), size=2)
        split_sets = []
        for pp in propotion[0]:
            split_sets.append(int(pp * len(trainset)))
        split_sets[-1] = len(trainset) - sum(split_sets[:-1])
    else:
        split_sets = (len(trainset) * (np.ones(args.TotalDevNum) / args.TotalDevNum)).astype(int)
    train_subsets = torch.utils.data.random_split(trainset, split_sets, generator=torch.Generator().manual_seed(args.seed))
    train_subset = train_subsets[args.DevNum - 1]
    trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    num_examples = {"trainset" : len(train_subset), "testset" : len(testset), "dm": data_mean, "ds": data_std}
    return trainloader, testloader, num_examples

def main(args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device, attacker, server_payload, secrets):

    if args.trained < 0 and args.attack == 'imprint':
        model, attacker, server_payload, secrets = attacks.Imprint_setting(args, model, loss_fn, setup)

    # print(model)
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

    # defense
    if args.defense == 'soteria':
        gt_gradient = defenses.defense_soteria(args,
            gt_imgs, gt_labels, model, loss_fn, device,
            layer_num=args.layer_num, percent_num=args.percent_num,
            perturb_imprint=args.perturb_imprint)
    elif args.defense == 'compression':
        gt_gradient = defenses.defense_compression(
            gt_gradients, device, percent_num=args.percent_num)
    elif args.defense == 'dp':
        gt_gradient = defenses.defense_dp(
            gt_gradients, device, loc=args.loc, scale=args.scale, noise_name=args.noise_name)
    elif args.defense == 'ours':  # optimize-based defense for all attacks
        gt_gradient, adv_imgs, adv_labels = defenses.defense_optim(
            args, model, loss_fn, gt_gradients, gt_imgs, gt_labels, dm, ds, device)
    else:
        print('No defenses!')

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
        output_denormalized = torch.clamp(reconstructed_data * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(gt_imgs * ds + dm, 0, 1)
        test_psnr = psnr(output_denormalized[-args.num_sen:], gt_denormalized[-args.num_sen:], batched=False, factor=1.)
        test_ssim = ssim_batch(output_denormalized[-args.num_sen:], gt_denormalized[-args.num_sen:])
        if args.dataset == 'ImageNet':
            test_lpips = lpips_loss(output_denormalized[-args.num_sen:].cpu(), gt_denormalized[-args.num_sen:].cpu())
        else:
            test_lpips = torch.tensor(-0.).to(device)
        print('PSNR {:.4f}, SSIM {:.4f}, LPIPS {:.4f}'.format(test_psnr.item(), test_ssim[0].item(), test_lpips.item()))
        # save
        if args.vis:
            rc_filename = args.defense + '_' + args.attack + '_psnr' + str(round(test_psnr.item(), 4)) + '.png'
            save_image(output_denormalized, os.path.join(args.output_dir, rc_filename))
            gt_filename = 'gt.png'
            save_image(gt_denormalized, os.path.join(args.output_dir, gt_filename))

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

    # Load Model
    attacker = None
    server_payload = None
    secrets = None
    loss_fn = nn.CrossEntropyLoss()
    if args.dataset == 'MNIST':
        if args.attack == 'imprint':
            model = LeNet_MNIST_imp()
            if args.trained > 0:
                model, attacker, server_payload, secrets = attacks.Imprint_setting(args, model, loss_fn, setup)
        else:
            model = LeNet_MNIST()
    elif args.dataset == 'CIFAR10':
        if args.attack == 'imprint':
            model = ConvNet_imp(width=32, num_classes=10, num_channels=3)
            if args.trained > 0:
                model, attacker, server_payload, secrets = attacks.Imprint_setting(args, model, loss_fn, setup)
        else:
            model = ConvNet(width=32, num_classes=10, num_channels=3)
    elif args.dataset == 'ImageNet':
        if args.attack == 'imprint':
            model = resnet18_imp(pretrained=args.pretrained)
            if args.trained > 0:
                model, attacker, server_payload, secrets = attacks.Imprint_setting(args, model, loss_fn, setup)
        else:
            model = resnet18(pretrained=args.pretrained)
    model.to(**setup)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.train_lr, momentum=0.9)

    # Load Dataset
    trainloader, testloader, num_examples = load_data(args)
    mean, std = num_examples["dm"], num_examples["ds"]
    dm = torch.as_tensor(mean, **setup)[None, :, None, None]  # 1xCx1x1
    ds = torch.as_tensor(std, **setup)[None, :, None, None]
    print('Total images {:d} on {}'.format(num_examples['trainset'], args.dataset))

    if args.vis:
        gt_imgs = []
        gt_labels = []
        for i, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            if i < args.batch_idx:
                continue

            # train n rounds
            if i < args.trained:
                model.train()
                optimizer_model.zero_grad()
                out, _, _ = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                optimizer_model.step()
                continue

            for j in range(labels.size(0)):
                if len(gt_imgs) == args.num_imgs:
                    break
                if labels[j] in gt_labels:
                    continue
                gt_labels.append(labels[j].unsqueeze(0))
                gt_imgs.append(imgs[j].unsqueeze(0))

            if len(gt_imgs) == args.num_imgs:
                break

        gt_labels = torch.cat(gt_labels)
        gt_imgs = torch.cat(gt_imgs)
        gt_imgs = gt_imgs.to(device)
        gt_labels = gt_labels.to(device)
        print('GT_labels: ', gt_labels.cpu())
        print('sensitive_labels: ', gt_labels[-args.num_sen:].cpu())
        print('Attack Method: ', args.attack, args.cost_fn)
        print('Defense Method: ', args.defense)

        start_time = time.time()
        test_psnr, test_ssim, test_lpips = main(
            args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device, attacker, server_payload, secrets)

    else:
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
