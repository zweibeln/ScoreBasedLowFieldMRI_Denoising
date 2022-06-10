from configs.imports import *
from losses.ssim_loss import *

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
                           'sigma_max': 378,  # Max std of noise
                           'dc': False,  # dc: data_consistency,
                           'img': int(6),  # img number to be reconstructed
                           'field_str_score': [1.5, 1.5],
                           'out_sz': (128, 128)
                },
                    sampling={
                            'steps': 5000,  # Num of Predictor Steps
                            'M': 1,  # Num of Corrector Steps
                            't_0': 1e-5,  # T min
                            'T': 1,  # T max
                            'snr': 0.16  # r=0.16
                },
                    training={
                            'lr': {'adam': 2e-4, 'sgd': 1e-4},
                            'epochs': 100,
                            'betas': (0.9, 0.999),
                            'milestones': [150000, 300000],
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
