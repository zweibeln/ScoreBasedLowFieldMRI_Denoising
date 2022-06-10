import sys
import os
ROOT = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.insert(0, ROOT)
from configs.imports import *
import torchvision.transforms as T
from configs.set_up_variables import *
from utils.snr_calculate import *
from time import ctime

def reorganize(img):
    sz = img.shape
    canvas = img.clone()
    canvas[:,:,0:int(sz[2]/2),0:int(sz[3]/2)]= img[:,:,int(sz[2]/2):sz[2],int(sz[3]/2):sz[3]]  # UL<-BR
    canvas[:,:,0:int(sz[2]/2),int(sz[3]/2):sz[3]]= img[:,:,int(sz[2]/2):sz[2],0:int(sz[3]/2)]  # UR<-BL
    canvas[:,:,int(sz[2]/2):sz[2],0:int(sz[3]/2)]= img[:,:,0:int(sz[2]/2),int(sz[3]/2):sz[3]]  # BL<-UR
    canvas[:,:,int(sz[2]/2):sz[2],int(sz[3]/2):sz[3]]= img[:,:,0:int(sz[2]/2),0:int(sz[3]/2)]  # BR<-UL
    return canvas


def add_noise(x_train,field_strength,frequency=True,return_std=0):
    '''
    :param x_train: torch.tensor(N,1,W,H)
    :param field_strength: float
    :param frequency: Boolean (True Default means acquisiton freq>5Mhz)
    :return: torch.tensor(N,1,W,H)
    '''
    x_train = x_train + torch.abs(torch.min(x_train))
    x_train = x_train/torch.max(x_train)
    # snr_init, sigma = 13.7278, 0.0143
    snr_init, sigma = 21.6732, 0.0094
    exponent = [7/4 if frequency else 3/2]  # freq>5Mhz -> 7/4 else 3/2
    snr_k = snr_init * pow((field_strength/1.5),exponent[0])  # SNR under x field strength
    sigma_a = (snr_init-snr_k)*sigma/snr_k  # New additional noise variance
    if sigma_a == 0:
        return x_train
    else:
        additional_noise_std = np.sqrt(sigma_a)  # new additional noise std
        n = torch.normal(mean=0, std=1, size=x_train.shape)*additional_noise_std
        # n_k = FT.fftn(n, s=None, dim=(-2,-1), norm=None)
        n_real = torch.normal(mean=0, std=1, size=x_train.shape)*additional_noise_std/np.sqrt(2)
        n_imag = torch.normal(mean=0, std=1, size=x_train.shape)*additional_noise_std/np.sqrt(2)
        n = torch.complex(n_real, n_imag)
        x_k = FT.fftn(x_train, s=None, dim=(-2, -1), norm='ortho')
        x_k = reorganize(x_k)
        x_k_noisy = x_k + n
        dims = x_k_noisy.shape
        resolution = (int(dims[-2]*(field_strength/1.5)),int(dims[-1]*(field_strength/1.5)))
        H = torch.zeros(x_k_noisy.shape)
        H[:, :, int((dims[-2]/2)-(resolution[0])/2):int((dims[-2]/2)+(resolution[0])/2),
                    int((dims[-1]/2)-(resolution[1])/2):int((dims[-1]/2)+(resolution[1])/2)] = 1
        x_k_low_res = x_k_noisy * H
        x_k_low_res = reorganize(x_k_low_res)
        # x_img_nosiy = torch.abs(FT.ifftn(x_k_low_res, s=None, dim=(-2, -1), norm='ortho'))
        x_img_nosiy = torch.real(FT.ifftn(x_k_low_res, s=None, dim=(-2, -1), norm='ortho'))
        x_img_nosiy = x_img_nosiy/torch.max(x_img_nosiy)
        # print(f'SNR BEFORE={snr_init,sigma}, SNR AFTER={snr_calculate(x_img_nosiy[0,0,:,:])[0]}, SNR DES={snr_k}')
        if return_std:
            return x_img_nosiy, H, (additional_noise_std+np.sqrt(sigma)) 
        else:
            return x_img_nosiy, H

def draw_data(field_strength=0.5, randomseed=42, set_type='val', num_volumes=6, return_std=0):
    data = torch.load(f'utils\knee_{set_type}.txt')
    random.seed(randomseed)
    volumes = random.sample(data,num_volumes) # Pick random 6 volumes
    observations = []
    masks = []
    targets = []
    noise_std = []
    for v in volumes:
        index = random.randint(0,25)
        targets.append(v[index].unsqueeze(0))
        if return_std:
            noisy_slice, mask, std = add_noise(v[index].unsqueeze(0), field_strength, True, return_std=return_std)
            observations.append(noisy_slice)
            masks.append(mask)
            noise_std.append(std)
        else:
            noisy_slice, mask = add_noise(v[index].unsqueeze(0), field_strength, True, return_std=return_std)
            observations.append(noisy_slice)
            masks.append(mask)
    if return_std:
        return observations, masks, targets, noise_std
    else:    
        return observations, masks, targets

def draw_data_train(set_type='train', low_f=0.5, high_f=0.7):
    data_path = os.getcwd() + '\\utils'
    # print(f'{data_path}')
    # sys.exit()
    volumes = torch.load(f'{data_path}\\knee_{set_type}.txt')
    observations = []
    masks = []
    targets = []
    print('TRAINING DATA IS LOADING ...')
    with tqdm.tqdm(total=len(volumes)) as pbar:
        for v in volumes:
            field_strength = np.random.uniform(low=low_f, high=high_f)
            targets.append(v)
            noisy_slice, mask = add_noise(v, field_strength, True)
            observations.append(noisy_slice)
            masks.append(mask)
            pbar.update(1)
    return observations, masks, targets
