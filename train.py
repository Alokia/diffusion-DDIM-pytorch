from dataset import create_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, load_yaml
import torch


def train(config):
    device = torch.device(config["device"])

    loader = create_dataset(**config["Dataset"])
    model = UNet(**config["Model"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    for epoch in range(1, config["epochs"] + 1):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch)
        torch.save(model.state_dict(), config["save_path"])


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)
