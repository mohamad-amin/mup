'''Train CIFAR10 with PyTorch.'''
import os
import math
import shutil
import argparse


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes

import resnet
from .. import example_utils


def coord_check(mup, lr, optimizer, nsteps, arch, base_shapes, nseeds, device='cuda', plotdir='', legend=False):

    optimizer = optimizer.replace('mu', '')

    def gen(w, standparam=False):
        def f():
            model = getattr(resnet, arch)(wm=w).to(device)
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, base_shapes)
            return model
        return f

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../dataset', train=True, download=True, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=False)

    widths = 2**np.arange(-2., 2)
    models = {w: gen(w, standparam=not mup) for w in widths}
    df = get_coord_data(models, dataloader, mup=mup, lr=lr, optimizer=optimizer, nseeds=nseeds, nsteps=nsteps)

    prm = 'μP' if mup else 'SP'
    plot_coord_data(df, legend=legend,
        save_to=os.path.join(plotdir, f'{prm.lower()}_{arch}_{optimizer}_coord.png'),
        suptitle=f'{prm} {arch} {optimizer} lr={lr} nseeds={nseeds}',
        face_color='xkcd:light grey' if not mup else None)


# Training
def train(epoch, net, use_progress_bar):
    from examples.ResNet.utils import progress_bar
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if use_progress_bar:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_acc = 100. * correct / total
    train_loss /= total
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, total, train_acc))

    return train_loss, train_acc


def test(epoch, net, use_progress_bar):
    from examples.ResNet.utils import progress_bar
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if use_progress_bar:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_acc = 100.*correct/total
    test_loss /= total

    # Save checkpoint.
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, total, test_acc))
    return test_loss, test_acc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=''
    '''
    PyTorch CIFAR10 Training, with μP.

    To save base shapes info, run e.g.

        python main.py --save_base_shapes resnet18.bsh --width_mult 1

    To train using MuAdam (or MuSGD), run

        python main.py --width_mult 2 --load_base_shapes resnet18.bsh --optimizer {muadam,musgd}

    To test coords, run

        python main.py --load_base_shapes resnet18.bsh --optimizer sgd --lr 0.1 --coord_check

        python main.py --load_base_shapes resnet18.bsh --optimizer adam --lr 0.001 --coord_check

    If you don't specify a base shape file, then you are using standard parametrization, e.g.

        python main.py --width_mult 2 --optimizer {muadam,musgd}

    Here muadam (resp. musgd) would have the same result as adam (resp. sgd).

    Note that models of different depths need separate `.bsh` files.
    ''', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config_path', default='', help="Path to a config file")
    parser.add_argument('--data_dir', type=str, default='/tmp')
    parser.add_argument('--save_dir', default='', help='Path to dir to save checkpoints and logs')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))
    config = example_utils.load_config(args.config_path)

    arch = config['model']['arch']
    width_mult = config['model'].get('width_mult', 1)
    growth_factor = config['model'].get('growth_factor', 2)
    save_base_shapes = config['model'].get('save_base_shapes', '')
    load_base_shapes = config['model'].get('load_base_shapes', '')
    do_coord_check = config['model'].get('coord_check', False)
    coord_check_nsteps = config['model']['coord_check_nsteps']
    coord_check_nseeds = config['model']['coord_check_nseeds']

    lr = config['train']['lr']
    resume = config['train']['resume']
    optimizer_name = config['train']['optimizer']
    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    test_batch_size = config['train']['test_batch_size']
    weight_decay = config['train']['weight_decay']
    num_workers = config['train']['num_workers']
    test_num_workers = config['train']['test_num_workers']
    momentum = config['train']['momentum']
    use_progress_bar = config['train'].get('use_progress_bar', False)

    train_size = config['data']['train_size']
    seed = config['data']['seed']

    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--resume', '-r', action='store_true',
    #                     help='resume from checkpoint')
    # parser.add_argument('--arch', type=str, default='resnet18')
    # parser.add_argument('--optimizer', default='musgd', choices=['sgd', 'adam', 'musgd', 'muadam'])
    # parser.add_argument('--epochs', type=int, default=150)
    # parser.add_argument('--width_mult', type=float, default=1)
    # parser.add_argument('--save_base_shapes', type=str, default='',
    #                     help='file location to save base shapes at')
    # parser.add_argument('--load_base_shapes', type=str, default='',
    #                     help='file location to load base shapes from')
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--test_batch_size', type=int, default=128)
    # parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--test_num_workers', type=int, default=2)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--data_dir', type=str, default='/tmp')
    # parser.add_argument('--train_size', type=int, default=-1)
    # parser.add_argument('--coord_check', action='store_true',
    #                     help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
    # parser.add_argument('--coord_check_nsteps', type=int, default=3,
    #                     help='Do coord check with this many steps.')
    # parser.add_argument('--coord_check_nseeds', type=int, default=1,
    #                     help='number of seeds for coord check')
    # parser.add_argument('--seed', type=int, default=1111,
    #                     help='random seed')
    # parser.add_argument('--run_index', type=int, default=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)

    # Data
    if not save_base_shapes:
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        if train_size != -1:
            perm = torch.randperm(len(trainset))
            idx = perm[:train_size]
            trainset = torch.utils.data.Subset(trainset, idx)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=test_num_workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

    if do_coord_check:
        print('testing parametrization')
        import os
        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        coord_check(mup=True,
            lr=lr, optimizer=optimizer_name, nsteps=coord_check_nsteps, arch=arch, base_shapes=load_base_shapes, nseeds=coord_check_nseeds, device=device, plotdir=plotdir, legend=False)
        coord_check(mup=False,
            lr=lr, optimizer=optimizer_name, nsteps=coord_check_nsteps, arch=arch, base_shapes=load_base_shapes, nseeds=coord_check_nseeds, device=device,plotdir=plotdir, legend=False)
        import sys; sys.exit()


    # Model
    print('==> Building model..')
    net = getattr(resnet, arch)(wm=width_mult, growth_factor=growth_factor)
    if save_base_shapes:
        print(f'saving base shapes at {save_base_shapes}')
        base_shapes = get_shapes(net)
        delta_shapes = get_shapes(getattr(resnet, arch)(wm=width_mult/2))
        make_base_shapes(base_shapes, delta_shapes, savefile=save_base_shapes)
        # save_shapes(net, save_base_shapes)
        print('done and exit')
        import sys; sys.exit()

    net = net.to(device)

    if load_base_shapes:
        print(f'loading base shapes from {load_base_shapes}')
        set_base_shapes(net, load_base_shapes)
        print('done')
    else:
        print(f'using standard parametrization')
        set_base_shapes(net, None)
        print('done')

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    def MSE_label(output, target):
        y_onehot = output.new_zeros(output.size(0), 10)
        y_onehot.scatter_(1, target.unsqueeze(-1), 1)
        y_onehot -= 1 / 10
        return F.mse_loss(output, y_onehot)

    criterion = MSE_label
    if optimizer_name == 'musgd':
        optimizer = MuSGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'muadam':
        optimizer = MuAdam(net.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logs = []
    for epoch in range(start_epoch, start_epoch+epochs):
        train_loss, train_acc = train(epoch, net, use_progress_bar)
        test_loss, test_acc = test(epoch, net, use_progress_bar)
        scheduler.step()
        logs.append(dict(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            width=width_mult,
            criterion='xent' if criterion.__name__ == 'cross_entropy' else 'mse',
        ))
        if math.isnan(train_loss):
            break

    with open(os.path.join(os.path.expanduser(args.save_dir), 'logs.tsv'), 'w') as f:
        logdf = pd.DataFrame(logs)
        print(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'))
        f.write(logdf.to_csv(sep='\t', float_format='%.4f'))
