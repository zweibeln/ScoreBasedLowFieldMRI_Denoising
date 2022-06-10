from configs.imports import *
from losses.score_loss import time_std
from configs.set_up_variables import sigma_max

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, initialize_weights=False, statement='enc'):
        '''
          inputs:
            in_ch  -> tuple : Input channel depths of the blocks in the network
            out_ch -> tuple : Output channel depths of the blokcs in the network
            initialize_weights -> boolean : Flag that selects the initialization type of the weights, True ->selects Kaiming weights, False -> Default weight initialization
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # self.drop = nn.Dropout2d(p=0.15)
        self.gn = nn.GroupNorm(32,out_ch)

        # This part takes care of the initialization of the weights if the initialize_weights flag is chosen
        if initialize_weights == True:
            for m in self.children():
                if type(m) == nn.Conv2d:
                    torch.nn.init.kaiming_normal_(m.weight,
                                                  nonlinearity='relu')  # Initialize Kernel weights of the Network by kaiming
                    torch.nn.init.zeros_(m.bias)  # Initialize biases as zero vectors

    def forward(self, x):
        # return self.gn(self.drop(self.relu(self.conv2(self.drop(self.relu(self.conv1(x)))))))
        return self.gn(self.relu(self.conv2(self.relu(self.conv1(x)))))
        


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), initialize_weights=False, statement='enc'):
        '''
        inputs:
          chs -> tuple : channel depths of the feature spaces in encoder side
          initialize_weights -> boolean : Flag that selects the initialization type of the weights, True ->selects Kaiming weights, False -> Default weight initialization
        '''
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], initialize_weights, statement=statement) for i in range(len(chs) - 1)])
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), initialize_weights=False, statement = 'dec'):
        '''
        inputs:
          chs -> tuple : channel depths of the feature spaces in decoder side
          initialize_weights -> boolean : Flag that selects the initialization type of the weights, True ->selects Kaiming weights, False -> Default weight initialization
        '''
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], initialize_weights, statement=statement) for i in range(len(chs) - 1)])
        for p in self.upconvs:
            if type(p) == nn.ConvTranspose2d:
                torch.nn.init.kaiming_normal_(p.weight,
                                              nonlinearity='relu')  # Initialize Kernel weights of the Network by kaiming
                torch.nn.init.zeros_(p.bias)  # Initialize biases as zero vectors

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNET_WOT(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1,
                 retain_dim=False, out_sz=(640, 368), initialize_weights=False, set_type='mri'):
        '''
        inputs:
          en_chs -> tuple: channel depths of the feature spaces in encoder side
          dec_chs -> tuple: channel depths of the feature spaces in decoder side
          num_class -> int: number of classes
          retain_dim -> boolean: True if the spatial dimension of the output is desired to be preserved
          out_sz -> tuple: spatial dimensions of the output
          initialize_weights -> boolean: Flag that selects the initialization type of the weights, True ->selects Kaiming weights, False -> Default weight initialization
        '''
        super().__init__()
        self.out_sz = out_sz
        self.encoder = Encoder(enc_chs, initialize_weights)
        self.decoder = Decoder(dec_chs, initialize_weights)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.sigma_max = sigma_max
        print(f'CHOSEN SIGMA MAX = {self.sigma_max}')
        
        # This part takes care of the initialization of the weights if the initialize_weights flag chosen
        if initialize_weights == True:
            torch.nn.init.kaiming_normal_(self.head.weight,
                                          nonlinearity='relu')  # Initialize Kernel weights of the Network by kaiming
            torch.nn.init.zeros_(self.head.bias)

    def forward(self, x, t):
        normalization_std = time_std(t,sigma_max=self.sigma_max)
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        
        return out / normalization_std