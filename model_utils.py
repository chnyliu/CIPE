import numpy as np


class Callback:
    def __init__(self): pass
    def on_train_begin(self, *args, **kwargs): pass
    def on_train_end(self, *args, **kwargs): pass
    def on_epoch_begin(self, *args, **kwargs): pass
    def on_epoch_end(self, *args, **kwargs): pass
    def on_batch_begin(self, *args, **kwargs): pass
    def on_batch_end(self, *args, **kwargs): pass
    def on_loss_begin(self, *args, **kwargs): pass
    def on_loss_end(self, *args, **kwargs): pass
    def on_step_begin(self, *args, **kwargs): pass
    def on_step_end(self, *args, **kwargs): pass


class EarlyStoppingLoss(Callback):
    def __init__(self, patience=30, tol=0.001, min_epochs=200):
        super(EarlyStoppingLoss, self).__init__()
        self.patience = patience
        self.tol = tol
        self.best = np.inf
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, metric, epoch_loss):
        metric = max(0, metric - self.tol)

        if metric < self.best:
            self.best = min(metric + self.tol, self.best)
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch > self.min_epochs:
                self.stopped_epoch = epoch
                return True
        return False