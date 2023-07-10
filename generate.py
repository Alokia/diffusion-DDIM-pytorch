from utils.engine import DDPMSampler, DDIMSampler
from model.UNet import UNet
import torch
from utils.tools import save_sample_image, save_image
from argparse import ArgumentParser


def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16)

    # sampler param
    parser.add_argument("--result_only", default=False, action="store_true")
    parser.add_argument("--interval", type=int, default=50)

    # DDIM sampler param
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # save image param
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    parser.add_argument("--to_grayscale", default=False, action="store_true")

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate(args):
    device = torch.device(args.device)

    cp = torch.load(args.checkpoint_path)
    # load trained model
    model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model = model.eval()

    if args.sampler == "ddim":
        sampler = DDIMSampler(model, **cp["config"]["Trainer"]).to(device)
    elif args.sampler == "ddpm":
        sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # generate Gaussian noise
    z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"],
                       *cp["config"]["Dataset"]["image_size"]), device=device)

    extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
    x = sampler(z_t, only_return_x_0=args.result_only, interval=args.interval, **extra_param)

    if args.result_only:
        save_image(x, nrow=args.nrow, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    else:
        save_sample_image(x, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)


if __name__ == "__main__":
    args = parse_option()
    generate(args)
