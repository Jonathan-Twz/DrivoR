"""Training-only removal of velocity/acceleration, preserving pose and command."""
import torch
from torch import nn


class EgoMotionDropout(nn.Module):
    def __init__(self, dim, probability=0.0, warmup_epochs=0):
        super().__init__()
        if not 0 <= probability <= 1 or warmup_epochs < 0:
            raise ValueError("Invalid ego-motion dropout settings")
        self.probability = float(probability)
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = 0
        self.missing_motion = nn.Parameter(torch.zeros(dim))
        self.last_drop_rate = None

    def forward(self, status, encoder):
        # Status order: pose[0:3], velocity[3:5], acceleration[5:7], command[7:11].
        encoded = encoder(status)
        self.last_drop_rate = status.new_zeros(())
        if not self.training or self.probability == 0 or self.epoch < self.warmup_epochs:
            return encoded
        dropped = torch.rand(status.shape[0], 1, device=status.device) < self.probability
        visible = status.clone().reshape(status.shape[0], -1, 11)
        visible[..., 3:7] = 0
        # Explicit learned missing-motion embedding, rather than treating missing as stationary.
        masked = encoder(visible.reshape_as(status)) + self.missing_motion
        self.last_drop_rate = dropped.float().mean()
        return torch.where(dropped, masked, encoded)
