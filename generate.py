from utils.engine import GaussianDiffusionSampler
from model.UNet import UNet
import torch
from utils.tools import save_sample_image, save_image
from argparse import ArgumentParser
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, default="./checkpoint/unet.pt")
    parser.add_argument("--device", type=str, default="cuda")

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, nargs="*", default=(32, 32))

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

    # sampler param
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta", type=float, nargs="*", default=(1e-4, 0.02))
    parser.add_argument("--result_only", default=False, action="store_true")
    parser.add_argument("--interval", type=int, default=50)

    # save image param
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default="./save/result.png")
    parser.add_argument("--to_grayscale", default=False, action="store_true")

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate(args):
    device = torch.device(args.device)

    # load trained model
    model = UNet(args.in_channels, args.model_channels, args.out_channels,
                 args.num_res_blocks, args.attention_resolutions, args.dropout,
                 args.channel_mult, args.conv_resample, args.num_heads)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)

    sampler = GaussianDiffusionSampler(model, args.beta, args.T).to(device)

    # generate Gaussian noise
    z_t = torch.randn((args.batch_size, args.in_channels, *args.image_size), device=device)

    x = sampler(z_t, only_return_x_0=args.result_only, interval=args.interval)

    if args.result_only:
        save_image(x, nrow=args.nrow, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    else:
        save_sample_image(x, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)


if __name__ == "__main__":
    args = parse_option()
    generate(args)
