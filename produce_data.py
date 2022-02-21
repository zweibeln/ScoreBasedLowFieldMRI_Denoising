from imports import *
import torchvision.transforms as T
from set_up_variables import *
from snr_calculate import *


def add_noise(x_train,field_strength,frequency=True):
    '''
    :param x_train: torch.tensor(N,1,W,H)
    :param field_strength: float
    :param frequency: Boolean (True Default means acquisiton freq>5Mhz)
    :return: torch.tensor(N,1,W,H)
    '''
    x_train = x_train/torch.max(x_train)
    snr_init, sigma = 13.7278, 0.0143
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
        x_k_noisy = x_k + n
        dims = x_k_noisy.shape
        resolution = (int(dims[-2]*(field_strength/1.5)),int(dims[-1]*(field_strength/1.5)))
        H = torch.zeros(x_k_noisy.shape)
        H[:, :, 0:resolution[0], 0:resolution[0]] = 1
        H[:, :, int(dims[-2]-resolution[0]):dims[-2], 0:resolution[1]] = 1
        H[:, :, int(dims[-2]-resolution[0]):dims[-2], int(dims[-1]-resolution[1]):dims[-1]] = 1
        H[:, :, 0:resolution[0], int(dims[-1]-resolution[1]):dims[-1]] = 1
        x_k_low_res = x_k_noisy * H
        x_img_nosiy = torch.abs(FT.ifftn(x_k_low_res, s=None, dim=(-2, -1), norm='ortho'))
        x_img_nosiy = x_img_nosiy/torch.max(x_img_nosiy)
        # print(f'SNR BEFORE={snr_init,sigma}, SNR AFTER={snr_calculate(x_img_nosiy[0,0,:,:])[0]}, SNR DES={snr_k}')
        return x_img_nosiy, H


def load_data(data_path):
    # print('Before : RAM usage is {} MB'.format(get_ram_usage_MB()))
    # loop through all the data and read it
    for file in os.listdir(data_path):
        data = h5py.File(Path(data_path, file), 'r')
    # print('After: RAM usage is {} MB'.format(get_ram_usage_MB()))

    resizing_size = (128,128)
    y = torch.unsqueeze(torch.from_numpy(data['reconstruction_esc'][:]), 1)
    rs = T.Resize(size=resizing_size)
    y = rs(y)

    return y


def produce_data_train(field_str_tr=[1.5,1.5]):
    # data_path = r'\\tsclient\C\utku\Master\Internship&Graduation\singlecoil_train'  # For Training Set
    data_path = r'C:\utku\Master\Internship&Graduation\singlecoil_train'  # For Training Set
    y_train = load_data(data_path).unsqueeze(1)
    field_strengths = np.random.uniform(field_str_tr[0], field_str_tr[1], y_train.shape[0])
    y_train = [add_noise(y, f)[0] for y, f in zip(y_train, field_strengths)]

    if centralize_data:
        y_train_pixel_sum = 0
        for y in y_train:
            y_train_pixel_sum += y
        y_train_std = torch.std(y_train_pixel_sum/len(y_train))
        y_train_mean = torch.mean(y_train_pixel_sum)/len(y_train)
        y_train = [(y-torch.tensor(y_train_mean))/y_train_std for y in y_train]

    return y_train


def produce_data_sampling(field_str_tr=[1.5,1.5]):
    # data_path = r'\\tsclient\C\utku\Master\Internship&Graduation\singlecoil_train'  # For Training Set
    data_path = r'C:\utku\Master\Internship&Graduation\singlecoil_train'  # For Training Set
    target = list(load_data(data_path).unsqueeze(1))

    # if centralize_data:
    #     target_pixel_sum = 0
    #     for y in target:
    #         target_pixel_sum += y
    #     y_train_std = torch.std(target_pixel_sum/len(target))
    #     y_train_mean = torch.mean(target_pixel_sum)/len(target)
    #     target = [(y-torch.tensor(y_train_mean))/y_train_std for y in target]
    #
    # snr = [snr_calculate(y[0,0,:,:])[0] for y in target]
    # plt.figure()
    # plt.plot(snr)
    # print(snr)
    # sys.exit()
    field_strengths = np.random.uniform(field_str_tr[0], field_str_tr[1], len(target))
    observation = [add_noise(y, f)[0] for y, f in zip(target, field_strengths)]
    mask = [add_noise(y, f)[1] for y, f in zip(target, field_strengths)]
    # data_path = r'\\tsclient\C\utku\Master\Internship&Graduation\FirstArch_Unet\Internship_FirsArch_Unet' + '\singlecoil_val'  # Validation Set
    # data_path = r'C:\utku\Master\Internship&Graduation\FirstArch_Unet\Internship_FirsArch_Unet' + '\singlecoil_val'  # Validation Set
    # y_val = load_data(data_path).unsqueeze(1)
    # field_strengths = np.random.uniform(field_str_tr[0], field_str_tr[1], y_val.shape[0])
    # y_val = [add_noise(y, f)[0] for y, f in zip(y_val, field_strengths)]
    # h_val = [add_noise(y, f)[1] for y, f in zip(y_val, field_strengths)]

    # y_val_pixel_sum = 0
    # for y in y_val:
    #     y_val_pixel_sum += y
    # y_val_std = torch.std(y_val_pixel_sum / len(y_val))
    # y_val_mean = torch.mean(y_val_pixel_sum) / len(y_val)
    # y_val = [(y - torch.tensor(y_val_mean)) / y_val_std for y in y_val]

    return observation, mask, target
