import torch
import math
from pathlib2 import Path
from typing import Union, Dict


class EarlyStopping:
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 1):
        """
        Decide whether to terminate the training early according to the indicators to be monitored
        and the number of times, and terminate early when the monitoring exceeds the specified number of
        consecutive times without better.

        Parameters:
            monitor: the indicators to be monitored will only take effect if they are passed a `dict`.
            mode: mode for monitoring indicators, 'min' or 'max'.
            patience: maximum tolerance times.
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.__value = -math.inf if mode == 'max' else math.inf
        self.__times = 0

    def state_dict(self) -> dict:
        """
        save state for next load recovery.

        for example:
            ```
            torch.save(state_dict, path)
            ```
        """
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.patience,
            'value': self.__value,
            'times': self.__times
        }

    def load_state_dict(self, state_dict: dict):
        """
        load state
        """
        self.monitor = state_dict['monitor']
        self.mode = state_dict['mode']
        self.patience = state_dict['patience']
        self.__value = state_dict['value']
        self.__times = state_dict['times']

    def reset(self):
        """
        reset tolerance times
        """
        self.__times = 0

    def step(self, metrics: Union[Dict, int, float]) -> bool:
        """
        Parameters:
            metrics: dict contains `monitor` or a scalar

        Returns:
            bool. Returns True if early termination is required, otherwise returns False
        """
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]

        if (self.mode == 'min' and metrics <= self.__value) or (
                self.mode == 'max' and metrics >= self.__value):
            self.__value = metrics
            self.__times = 0
        else:
            self.__times += 1
        if self.__times >= self.patience:
            return True
        return False


class ModelCheckpoint:
    def __init__(self, filepath: str = 'checkpoint.pth', monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = False, save_freq: int = 1):
        """
        auto save checkpoint during training.

        Parameters:
            filepath: File name or folder name, the location where it needs to be saved,
                or in the case of a folder, the number of checkpoints saved may be more than one.
            monitor: the indicators to be monitored will only take effect if they are passed a `dict`.
            mode: mode for monitoring indicators, 'min' or 'max'.
            save_best_only: whether to save only the checkpoints with the best indicators.
            save_freq: frequency of saving, only valid if `save_best_only=False`.
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.__times = 1
        self.__value = -math.inf if mode == 'max' else math.inf

    @staticmethod
    def save(filepath: str, times: int = None, **kwargs):
        """
        save checkpoint.

        Parameters:
            filepath: File name or folder name, the location where it needs to be saved,
                or in the case of a folder, the number of checkpoints saved may be more than one.
            times: number of current saves, used only if the save path is a folder, used only for naming files.
            kwargs: all content to be saved.
        """
        path = Path(filepath)
        if path.is_dir():
            if not path.exists():
                path.mkdir(parents=True)
            path.joinpath(f'checkpoint-{times}.pth')
        torch.save(kwargs, str(path))

    def state_dict(self):
        """
        save state for next load recovery.

        for example:
            ```
            torch.save(state_dict, path)
            ```
        """
        return {
            'filepath': self.filepath,
            'monitor': self.monitor,
            'save_best_only': self.save_best_only,
            'mode': self.mode,
            'save_freq': self.save_freq,
            'times': self.__times,
            'value': self.__value
        }

    def load_state_dict(self, state_dict: dict):
        """
        load state
        """
        self.filepath = state_dict['filepath']
        self.monitor = state_dict['monitor']
        self.save_best_only = state_dict['save_best_only']
        self.mode = state_dict['mode']
        self.save_freq = state_dict['save_freq']
        self.__times = state_dict['times']
        self.__value = state_dict['value']

    def reset(self):
        """
        reset count times
        """
        self.__times = 1

    def step(self, metrics: Union[Dict, int, float], **kwargs):
        """
        Parameters:
            metrics: dict contains `monitor` or a scalar
            kwargs: all content to be saved.
        """
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]

        flag = False

        if self.save_best_only:
            if (self.mode == 'min' and metrics <= self.__value) or (
                    self.mode == 'max' and metrics >= self.__value):
                self.__value = metrics
                self.save(self.filepath, self.__times, **kwargs)
                flag = True
        else:
            if self.__times % self.save_freq == 0:
                self.save(self.filepath, self.__times, **kwargs)
                flag = True

        self.__times += 1
        return flag
