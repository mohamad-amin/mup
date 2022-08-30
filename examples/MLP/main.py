import time
import os
import yaml
import shutil
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
import argparse
import math

from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def coord_check(mup, lr, train_loader, nsteps, nseeds, output_mult, input_mult, load_base_shapes, growth_factor, plotdir='', legend=False):

    def gen(w, standparam=False):
        def f():
            model = MLP(width=w, nonlin=torch.tanh, output_mult=output_mult, input_mult=input_mult, growth_factor=growth_factor).to(device)
            if standparam:
                set_base_shapes(model, None)
            else:
                assert load_base_shapes, 'load_base_shapes needs to be nonempty'
                set_base_shapes(model, load_base_shapes)
            return model
        return f

    widths = 2**np.arange(7, 14)
    models = {w: gen(w, standparam=not mup) for w in widths}

    df = get_coord_data(models, train_loader, mup=mup, lr=lr, optimizer='sgd', flatten_input=True, nseeds=nseeds, nsteps=nsteps, lossfn='nll')

    prm = 'μP' if mup else 'SP'
    return plot_coord_data(df, legend=legend,
        save_to=os.path.join(plotdir, f'{prm.lower()}_mlp_sgd_coord.png'),
        suptitle=f'{prm} MLP SGD lr={lr} nseeds={nseeds}',
        face_color='xkcd:light grey' if not mup else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    PyTorch MLP on CIFAR-10, with μP.

    This is the scripts we use in the MLP experiment in our paper.

    To train a μP model, one needs to first specify the base shapes. To save base shapes info, run, for example,

        python main.py --save_base_shapes width64.bsh

    To train using MuSGD, run

        python main.py --load_base_shapes width64.bsh

    To perform coord check, run

        python main.py --load_base_shapes width64.bsh --coord_check

    If you don't specify a base shape file, then you are using standard parametrization

        python main.py

    We provide below some optimal hyperparameters for different activation/loss function combos:
        if nonlin == torch.relu and criterion == F.cross_entropy:
            input_mult = 0.00390625
            output_mult = 32
        elif nonlin == torch.tanh and criterion == F.cross_entropy:
            input_mult = 0.125
            output_mult = 32
        elif nonlin == torch.relu and criterion == MSE_label:
            input_mult = 0.03125
            output_mult = 32
        elif nonlin == torch.tanh and criterion == MSE_label:
            input_mult = 8
            output_mult = 0.125
    ''', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config_path', default='', help="Path to a config file")
    parser.add_argument('--data_dir', type=str, default='/tmp')
    parser.add_argument('--save_dir', default='', help='Path to dir to save checkpoints and logs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))
    config = load_config(args.config_path)

    width_mult = config['model'].get('width_mult', 1)
    growth_factor = config['model'].get('growth_factor', 2)
    save_base_shapes = config['model'].get('save_base_shapes', '')
    load_base_shapes = config['model'].get('load_base_shapes', '')
    do_coord_check = config['model'].get('coord_check', False)
    coord_check_nsteps = config['model']['coord_check_nsteps']
    coord_check_nseeds = config['model']['coord_check_nseeds']
    output_mult = config['model'].get('output_mult', 1.0)
    input_mult = config['model'].get('input_mult', 1.0)
    init_std = config['model'].get('init_std', 1.0)

    lr = config['train']['lr']
    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']
    momentum = config['train']['momentum']
    criterion_name = config['train'].get('criterion', 'l2')
    no_shuffle = config['train'].get('no_shuffle', True)

    train_size = config['data']['train_size']
    seed = config['data']['seed']
    log_interval = config['data'].get('log_interval', 300)

    torch.manual_seed(seed)
    device = torch.device("cuda")
    kwargs = {'num_workers': 0, 'pin_memory': True}

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    if train_size != -1:
        perm = torch.randperm(len(trainset))
        idx = perm[:train_size]
        trainset = torch.utils.data.Subset(trainset, idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=not no_shuffle, num_workers=0, )

    testset = datasets.CIFAR10(root=args.data_dir, train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    class MLP(nn.Module):
        def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0, growth_factor=1):
            super(MLP, self).__init__()
            self.nonlin = nonlin
            self.input_mult = input_mult
            self.output_mult = output_mult
            self.fc_1 = nn.Linear(3072, width, bias=False)
            self.fc_2 = nn.Linear(width, int(width * growth_factor), bias=False)
            self.fc_3 = MuReadout(int(width * growth_factor), num_classes, bias=False, output_mult=output_mult)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
            self.fc_1.weight.data /= self.input_mult**0.5
            self.fc_1.weight.data *= init_std
            nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
            self.fc_2.weight.data *= init_std
            nn.init.zeros_(self.fc_3.weight)

        def forward(self, x):
            out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
            out = self.nonlin(self.fc_2(out))
            return self.fc_3(out)


    def train(model, device, train_loader, optimizer, epoch,
            scheduler=None, criterion=F.cross_entropy):
        model.train()
        train_loss = 0
        correct = 0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item() * data.shape[0]  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ms/batch {:5.2f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    elapsed * 1000 / log_interval))
                start_time = time.time()
            if scheduler is not None:
                scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        return train_loss, train_acc

    def test(model, device, test_loader,
            evalmode=True, criterion=F.cross_entropy):
        if evalmode:
            model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data.view(data.size(0), -1))
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss, correct / len(test_loader.dataset)


    def MSE_label(output, target):
        y_onehot = output.new_zeros(output.size(0), 10)
        y_onehot.scatter_(1, target.unsqueeze(-1), 1)
        y_onehot -= 1/10
        return F.mse_loss(output, y_onehot)
    
    if do_coord_check:
        print('testing parametrization')
        import os
        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        coord_check(mup=True, lr=lr, train_loader=train_loader, nsteps=coord_check_nsteps, nseeds=coord_check_nseeds, output_mult=output_mult, input_mult=input_mult, load_base_shapes=load_base_shapes, growth_factor=growth_factor, plotdir=plotdir, legend=False)
        coord_check(mup=False, lr=lr, train_loader=train_loader, nsteps=coord_check_nsteps, nseeds=coord_check_nseeds, output_mult=output_mult, input_mult=input_mult, load_base_shapes=load_base_shapes, growth_factor=growth_factor, plotdir=plotdir, legend=False)
        import sys; sys.exit()

    logs = []
    # for nonlin in [torch.relu, torch.tanh]:
    for nonlin in [torch.relu]:
        # for criterion in [F.cross_entropy, MSE_label]:
        for criterion in [MSE_label if criterion_name == 'l2' else F.cross_entropy]:

            for width in [int(64 * width_mult)]:
                # print(f'{nonlin.__name__}_{criterion.__name__}_{str(width)}')
                mynet = MLP(width=width, nonlin=nonlin, output_mult=output_mult, input_mult=input_mult, growth_factor=growth_factor).to(device)
                if save_base_shapes:
                    print(f'saving base shapes at {save_base_shapes}')
                    base_shapes = get_shapes(mynet)
                    delta_shapes = get_shapes(
                        # just need to change whatever dimension(s) we are scaling
                        MLP(width=width+1, nonlin=nonlin, output_mult=output_mult, input_mult=input_mult, growth_factor=growth_factor)
                    )
                    make_base_shapes(base_shapes, delta_shapes, savefile=save_base_shapes)
                    print('done and exit')
                    import sys; sys.exit()
                if load_base_shapes:
                    print(f'loading base shapes from {load_base_shapes}')
                    set_base_shapes(mynet, load_base_shapes)
                    print('done')
                else:
                    print(f'using own shapes')
                    set_base_shapes(mynet, None)
                    print('done')
                print('Width:', width)
                optimizer = MuSGD(mynet.parameters(), lr=lr, momentum=momentum)
                for epoch in range(1, epochs+1):
                    train_loss, train_acc, = train(mynet, device, train_loader, optimizer, epoch, criterion=criterion)
                    if epoch - 1 > 0.8 * epochs:
                        test_loss, test_acc = test(mynet, device, test_loader)
                    else:
                        test_loss, test_acc = None, None
                    logs.append(dict(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        test_loss=test_loss,
                        test_acc=test_acc,
                        width=width,
                        nonlin=nonlin.__name__,
                        criterion='xent' if criterion.__name__=='cross_entropy' else 'mse',
                    ))
                    if math.isnan(train_loss):
                        break
    
    with open(os.path.join(os.path.expanduser(args.save_dir), 'logs.tsv'), 'w') as f:
        logdf = pd.DataFrame(logs)
        print(os.path.join(os.path.expanduser(args.save_dir), 'logs.tsv'))
        f.write(logdf.to_csv(sep='\t', float_format='%.4f'))
