from dataset.MNIST import create_mnist_dataset
from dataset.CIFAR import create_cifar10_dataset
from dataset.Custom import create_custom_dataset


def create_dataset(dataset: str, **kwargs):
    if dataset == "mnist":
        return create_mnist_dataset(**kwargs)
    elif dataset == "cifar":
        return create_cifar10_dataset(**kwargs)
    elif dataset == "custom":
        return create_custom_dataset(**kwargs)
    else:
        raise ValueError(f"dataset except one of {'mnist', 'cifar', 'custom'}, but got {dataset}")
