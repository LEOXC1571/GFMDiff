
import torch
import torch.nn as nn
import numpy as np


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s

    return alphas2


def sigma(gamma):
    return torch.sqrt(torch.sigmoid(gamma))


def alpha(gamma):
    return torch.sqrt(torch.sigmoid(-gamma))


def snr(gamma):
    return torch.exp(-gamma)


class NoiseSchedule(nn.Module):
    def __init__(self, noise_schedule, timestep, precision):
        super(NoiseSchedule, self).__init__()
        self.noise_schedule = noise_schedule
        self.timestep = timestep
        self.precision = precision
        if 'polynomial' in noise_schedule:
            power = float(noise_schedule.split('_')[1])
            alphas2 = polynomial_schedule(timestep, s=precision, power=power)
        elif noise_schedule == 'sigmoid':
            betas = np.linspace(-6, 6, timestep+1)
            betas = sigmoid(betas)
            alpahs = np.cumprod((1. - betas), axis=0)
            alphas2 = alpahs ** 2
        sigmas2 = 1 - alphas2
        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2
        self.gamma = torch.nn.Parameter(torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timestep).long()
        return self.gamma[t_int]
