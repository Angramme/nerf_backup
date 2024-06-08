# realnvp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RealNVP(nn.Module):
    def __init__(self, num_scales, in_channels, mid_channels, num_blocks):
        super(RealNVP, self).__init__()
        self.num_scales = num_scales
        self.transforms = nn.ModuleList()
        for _ in range(num_scales):
            self.transforms.append(RealNVPBlock(in_channels, mid_channels, num_blocks))

    def forward(self, x, reverse=False):
        log_det_jacobian = 0
        for transform in (self.transforms if not reverse else reversed(self.transforms)):
            x, log_det_jacobian = transform(x, reverse, log_det_jacobian)
        return x, log_det_jacobian

    def log_prob(self, x):
        z, log_det_jacobian = self(x)
        log_prob_z = Normal(0, 1).log_prob(z).sum([1, 2, 3])
        return log_prob_z + log_det_jacobian

class RealNVPBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks):
        super(RealNVPBlock, self).__init__()
        self.s = nn.ModuleList()
        self.t = nn.ModuleList()
        for _ in range(num_blocks):
            self.s.append(self._create_net(in_channels, mid_channels))
            self.t.append(self._create_net(in_channels, mid_channels))

    def forward(self, x, reverse=False, log_det_jacobian=0):
        if not reverse:
            return self._forward(x, log_det_jacobian)
        else:
            return self._reverse(x, log_det_jacobian)

    def _forward(self, x, log_det_jacobian):
        for s, t in zip(self.s, self.t):
            x1, x2 = x.chunk(2, 1)
            s_out = s(x1)
            t_out = t(x1)
            z1 = x1
            z2 = x2 * torch.exp(s_out) + t_out
            x = torch.cat([z1, z2], 1)
            log_det_jacobian += s_out.sum([1, 2, 3])
        return x, log_det_jacobian

    def _reverse(self, z, log_det_jacobian):
        for s, t in zip(reversed(self.s), reversed(self.t)):
            z1, z2 = z.chunk(2, 1)
            s_out = s(z1)
            t_out = t(z1)
            x1 = z1
            x2 = (z2 - t_out) * torch.exp(-s_out)
            z = torch.cat([x1, x2], 1)
            log_det_jacobian -= s_out.sum([1, 2, 3])
        return z, log_det_jacobian

    def _create_net(self, in_channels, mid_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels // 2, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels // 2, 3, padding=1)
        )
