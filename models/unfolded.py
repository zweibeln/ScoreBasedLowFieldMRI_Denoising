from configs.imports import *
from configs.set_up_variables import *
from utils.draw_data import reorganize
from models.unet import *


class Unfolded(nn.Module):
    def __init__(self, iteration=4):
        super().__init__()
        self.mu = nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1) for _ in range(iteration)])
        # self.proximal = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
        #                                              nn.ReLU(),
        #                                              nn.MaxPool2d(2),
        #                                              nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
        #                                              nn.ReLU(),
        #                                              nn.MaxPool2d(2),
        #                                              nn.ConvTranspose2d(in_channels=64, out_channels=64,kernel_size=2,stride=2),
        #                                              nn.ReLU(),
        #                                              nn.ConvTranspose2d(in_channels=64, out_channels=64,kernel_size=2,stride=2),
        #                                              nn.ReLU(),
        #                                              nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,padding=1),
        #                                              nn.Sigmoid())
        #                                for _ in range(iteration)])
        self.proximal = nn.ModuleList([UNET(enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1,
                 retain_dim=False, out_sz=(128, 128), initialize_weights=False, set_type='mri') for _ in range(iteration)])


    def forward(self, x, mask, y):  # y->obsevation
        for mu,proximal in zip(self.mu,self.proximal):
            dc_f = mask*(y-reorganize(FT.fftn(x)))
            dc_t = torch.abs(FT.ifftn(reorganize(dc_f), s=None, dim=(-2,-1), norm=None))
            x_dc = x.to(device) + mu(dc_t.to(device))
            x = proximal(x_dc).to('cpu')
        return x
