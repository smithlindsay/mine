import numpy as np
import math
import torch
import torch.nn as nn
import mine.helpers as utils
from tqdm.auto import tqdm
# import time

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
datadir = "/scratch/gpfs/ls1546/mine/data/"

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = torch.logsumexp(input, 0) - math.log(input.shape[0])
        
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (grad_output.log() + input.detach() - math.log(input.shape[0]) - (running_mean + EPS).log()).exp()
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema
    return t_log, running_mean

class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, device=device):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.device = device
        self.T = T.to(device)

    # if the values from T(x, z_marg) are too large to fit in float32 when exponentiated, 
    # the loss will be nan. toggle the biased calculation on instead
    def bias_toggle(self, t_marg):
        if (torch.max(t_marg) - math.log(t_marg.shape[0])) > 87:
            self.loss = 'mine_biased'
        else:
            self.loss = 'mine'

    def checkpoint(self, curr_loss, best_loss, name, count):
        threshold = 0.005
        if curr_loss < threshold:
            if curr_loss < best_loss:
                best_loss = curr_loss
                torch.save(self.T, f"{datadir}{name}_best_mine.pth")
                np.save(f"{datadir}{name}_best_loss.npy", best_loss)
            
            torch.save(self.T, f"{datadir}{name}_ckpt_mine{count}.pth")
            count += 1
        return best_loss, count
    
    def avg_checkpoint(self, name, count):
        if count > 1:
            model = torch.load(f"{datadir}{name}_ckpt_mine0.pth")
            weights = model.fc1x.weight.detach().cpu().numpy()[0]
            avg_weights = weights
            for i in range(1, count):
                model = torch.load(f"{datadir}{name}_ckpt_mine{i}.pth")
                weights = model.fc1x.weight.detach().cpu().numpy()[0]
                avg_weights = avg_weights + weights
            avg_weights = avg_weights / count
            # model.fc1x.weight = nn.Parameter(torch.tensor(avg_weights))
            np.save(f"{datadir}avg_weights_{name}.npy", avg_weights)

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        x = x.to(device)
        z = z.to(device)
        z_marg = z_marg.to(device)

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)
        self.bias_toggle(t_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
            
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, epochs, batch_size, lam, name, X_test, Y_test, opt=None):
        count = 0
        best_loss = np.inf
        losses = []
        loss_type = []

        if opt is None:
            opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=lam)

        dataloader =  utils.dataloader(X, Y, batch_size)

        for epoch in (pbar := tqdm(range(1, epochs + 1))):
            mu_mi = 0
            # end = time.time()
            for x, y in dataloader:
                # measure data loading time
                # data_time = time.time() - end

                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = self.forward(x, y)
                # save 0 for biased, 1 for unbiased
                if self.loss == 'mine_biased':
                    loss_type.append(0)
                else:
                    loss_type.append(1)
                losses.append(-loss.item())
                loss.backward()
                opt.step()

                mu_mi -= loss.item()
                # measure elapsed time
                # batch_time = time.time() - end
                # end = time.time()
                # print(f"batch time: {batch_time}, data time: {data_time}")

            pbar.set_description(f"epoch: {epoch}, mu_mi: {mu_mi:4f}")
            curr_loss = losses[epoch]
            # checkpoint the model if loss below threshold & save best model
            best_loss, count = self.checkpoint(curr_loss, best_loss, name, count)
        final_mi = self.mi(X_test, Y_test)
        print(f"Final MI on test data: {final_mi.item()}")

        # Average the weights of the checkpointed models
        self.avg_checkpoint(name, count)

        return final_mi, losses, loss_type