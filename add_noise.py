from imports import *


def add_noise(x_train,field_strength,frequency=True):
    '''
    :param x_train: torch.tensor(N,1,W,H)
    :param field_strength: float
    :param frequency: Boolean (True Default means acquisiton freq>5Mhz)
    :return: torch.tensor(N,1,W,H)
    '''

    # This function adds Gaussian noise to k-space data according to given field strength and frequency
    sigma = 1.0821552e-08  # Noise variance for 1.5T field str
    snr_init = 18.19766  # SNR under 1.5T field strength
    exponent = [7/4 if frequency else 3/2]  # freq>5Mhz -> 7/4 else 3/2
    snr_k = snr_init * pow((field_strength/1.5),exponent[0])  # SNR under x field strength
    sigma_a = (snr_init-snr_k)*sigma/snr_k  # New additional noise variance
    if sigma_a == 0:
        return x_train
    else:
        additional_noise_std = np.sqrt(sigma_a)  # new additional noise std
        n_real = torch.normal(mean=0, std=1, size=x_train.shape)*additional_noise_std/np.sqrt(2)
        n_imag = torch.normal(mean=0, std=1, size=x_train.shape)*additional_noise_std/np.sqrt(2)
        n = torch.complex(n_real, n_imag)
        return x_train + n
