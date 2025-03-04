# %%
from pathlib import Path
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if specific metric loss doesn't improve after a given patience."""

    def __init__(
        self,
        metric,
        mode,
        patience=0,
        verbose=False,
        delta=0,
        path = '/home/bkchoi/tomocube_preprocess_yang/bkchoi/model/',
        trace_func=print,
    ):

        self.metric = metric
        if mode == "min":
            self.metric_best = np.Inf
        elif mode == "max":
            self.metric_best = -np.Inf
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, epoch, metric, model):
        self.epoch = epoch
        if self.mode == "min":
            score = -metric
        elif self.mode == "max":
            score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        """Saves model when metric improves."""
        descript_dict = {"min": "decreased", "max": "increased"}
        if self.verbose:
            self.trace_func(
                f"Metric({self.metric}) {descript_dict[self.mode]} ({self.metric_best:.6f} --> {metric:.6f}).  Saving model ..."
            )
        #torch.save(model.state_dict(), Path(self.path, Path(f'{self.epoch}_ckpt.pt')))
        torch.save(model.state_dict(), Path('/home/bkchoi/tomocube_preprocess_yang/bkchoi/model/', Path(f'{self.epoch}_ckpt.pt')))
        self.metric_best = metric


