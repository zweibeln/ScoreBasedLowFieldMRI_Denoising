from imports import *
from ssim_loss import SSIM_loss


class ConfigObject:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [ConfigObject(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, ConfigObject(val) if isinstance(val, dict) else val)


config_dict = dict(mri={
                           'sigma_min': 0.01,  # Min std of noise
                           'sigma_max': 8.076,  # Max std of noise
                           'dc': True,  # dc: data_consistency,
                           'img': int(15),  # img number to be reconstructed
                           'field_str_score': [1.5, 1.5],
                           'out_sz': (128, 128)
                },
                    mnist={
                            'sigma_min': 0.01,  # Min std of noise
                            'sigma_max': 3.4,  # Max std of noise
                            'dc': False,  # dc: data consistency
                            'img': np.random.permutation(range(64)),  # number of images in a patch to be recovered
                            'out_sz': (32, 32)
                },
                    sampling={
                            'langevin': True,
                            'steps': 500,  # Num of Predictor Steps
                            'M': 1,  # Num of Corrector Steps
                            't_0': 1e-5,  # T min
                            'T': 1,  # T max
                            'snr': 0.16,  # r=0.16
                            'dc_loss': {
                                 'ssim': (SSIM_loss(), 'ssim'),  # SSIM loss for data consistency
                                 'mse': (nn.MSELoss(), 'mse')  # MSE loss for data consistency
                            },
                            'dc_lr': 1e-4,  # Data Consistency Step learning rate
                            'dc_f': [0.5, 0.5],  # MRI field strength range
                            'dc_img_patch': 9
                },
                    training={
                            'lr': {'adam': 2e-4, 'sgd': 1e-4},
                            'epochs': 2,
                            'betas': (0.9, 0.999),
                            'milestones': [1500, 3000],
                            'gamma': [0.5],
                            'ema': 0.999,
                            'grad_clip_norm': 1,
                            'centralize_data': True,
                            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                },
                    model={
                            'model_name': 'unet_wot',
                            'retain_dim': True,
                            'initialize_weights': True,
                            'num_class': 1,
                            'enc_chs': (1, 64, 128, 256, 512, 1024),
                            'dec_chs': (1024, 512, 256, 128, 64),
                },

                    set_type='mri',

                )

config = ConfigObject(config_dict)
