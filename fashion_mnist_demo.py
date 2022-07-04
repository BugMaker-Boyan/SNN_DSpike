import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from spikingjelly.activation_based import neuron, functional, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing
from DSpikeSG import DSpike


class SCNN(nn.Module):
    def __init__(self, T, channels, use_cupy):
        super(SCNN, self).__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=DSpike(alpha=3.)),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=DSpike(alpha=3.)),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=DSpike(alpha=3.)),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=DSpike(alpha=3.)),
        )

        functional.set_step_mode(self, 'm')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

    def spiking_encoder(self):
        return self.conv_fc[0:3]


def main():
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=100, type=int)
    parser.add_argument('-device', default='cuda:0')
    parser.add_argument('-b', default=64, type=int)
    parser.add_argument('-epochs', default=100, type=int)
    parser.add_argument('-j', default=4, type=int)
    parser.add_argument('-data-dir', type=str)
    parser.add_argument('-out-dir', type=str, default='./logs')
    parser.add_argument('-resume', type=str)
    parser.add_argument('-amp', action='store_true')
    parser.add_argument('-cupy', action='store_true')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'])
    parser.add_argument('-momentum', default=0.9, type=float)
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-channels', default=128, type=int)
    parser.add_argument('-save-es', default=None)

    args = parser.parse_args()
    print(args)

    net = SCNN(T=args.T, channels=args.channels, use_cupy=args.cupy)

    print(net)

    net.to(args.device)

    train_dataset = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        if args.save_es is not None and args.save_es != '':
            encoder = net.spiking_encoder()
            with torch.no_grad():
                for img, label in test_data_loader:
                    img = img.to(args.device)
                    label = label.to(args.device)
                    # img.shape = [N, C, H, W]
                    img_seq = img.unsqueeze(0).repeat(net.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
                    spike_seq = encoder(img_seq)
                    functional.reset_net(encoder)
                    to_pil_img = torchvision.transforms.ToPILImage()
                    vs_dir = os.path.join(args.save_es, 'visualization')
                    os.mkdir(vs_dir)

                    img = img.cpu()
                    spike_seq = spike_seq.cpu()

                    img = F.interpolate(img, scale_factor=4, mode='bilinear')
                    # 28 * 28 is too small to read. So, we interpolate it to a larger size

                    for i in range(label.shape[0]):
                        vs_dir_i = os.path.join(vs_dir, f'{i}')
                        os.mkdir(vs_dir_i)
                        to_pil_img(img[i]).save(os.path.join(vs_dir_i, f'input.png'))
                        for t in range(net.T):
                            print(f'saving {i}-th sample with t={t}...')
                            # spike_seq.shape = [T, N, C, H, W]

                            visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.png'), pad_inches=0.02)
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.pdf'), pad_inches=0.02)
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.svg'), pad_inches=0.02)
                            plt.clf()

                    exit()

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')


    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))


    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()

                functional.reset_net(net)
            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss = {train_loss: .4f}, '
            f'train_acc = {train_acc: .4f}, test_loss = {test_loss: .4f}, test_acc = {test_acc: .4f}, '
            f'max_test_acc = {max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
