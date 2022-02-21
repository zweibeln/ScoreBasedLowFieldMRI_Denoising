from imports import *


class DC_StepSize(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        nn.init.normal_(self.conv.weight.data, mean=0, std=1e-4)

    def forward(self, x):
        return self.conv(x)
