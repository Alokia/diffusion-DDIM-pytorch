from dataset.MNIST import create_mnist_dataset
from dataset.CIFAR import create_cifar100_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch
from argparse import ArgumentParser
import torch


def parse_option():
    parser = ArgumentParser()

    # train param
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save_path", type=str, default="./checkpoint/unet.pt")
    parser.add_argument("--consume", default=False, action="store_true")
    parser.add_argument("--consume_path", type=str, default="./checkpoint/unet.pt")

    # dataset param
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("--download", default=False, action="store_true")

    # UNet param
    parser.add_argument("-ic", "--in_channels", type=int, default=3)
    parser.add_argument("-mc", "--model_channels", type=int, default=128)
    parser.add_argument("-oc", "--out_channels", type=int, default=3)
    parser.add_argument("-ar", "--attention_resolutions", type=int, nargs="*", default=tuple())
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--channel_mult", type=int, nargs="*", default=(1, 2, 2, 2))
    parser.add_argument("--conv_resample", action="store_true", default=False)
    parser.add_argument("--num_heads", type=int, default=4)

    # trainer param
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta", type=float, nargs="*", default=(1e-4, 0.02))

    # optimizer param
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()
    return args


def train(args):
    device = torch.device(args.device)

    loader = create_mnist_dataset(args.data_path, args.batch_size, download=args.download)
    model = UNet(args.in_channels, args.model_channels, args.out_channels,
                 args.num_res_blocks, args.attention_resolutions, args.dropout,
                 args.channel_mult, args.conv_resample, args.num_heads).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, args.beta, args.T).to(device)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch)
        torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    args = parse_option()
    print(args)
    train(args)
