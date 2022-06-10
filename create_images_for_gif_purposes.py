from configs.imports import *
from models.init_model import *
from configs.set_up_variables import *
from plot_images import *
from configs.print_configs import *
from PIL import Image
import torchvision.transforms.functional as TF
from models.unfolded import *
from models.unet import *
from utils.draw_data import *


# LOADING TRAINED MODELS
score_model, _,_,_,_ = init_model()

plt.style.use('grayscale')
# LOADING DATA SET
f_str = 0.5
# observation, mask_, target_ = produce_data_sampling2(field_str_tr=[f_str,f_str], resizing=(128,128), set='train')
randomseed=104
set_type = 'val'
obs_, mask_, target_ = draw_data(f_str,randomseed=randomseed, set_type=set_type, num_volumes=1)
cnt=0
for idx,(obs,mask,target) in enumerate(zip(obs_,mask_,target_)):
    ########### STANDARDIZING THE IMAGES ###############
    obs = obs - torch.min(obs)
    obs = obs/torch.max(obs)
    tar = target - torch.min(target)
    tar = tar/torch.max(tar)

    # INFERENCE
    ##################################### SDE ####################################################
    torch.set_grad_enabled(False)
    mask_complement = 1-mask
    x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    dst = np.sqrt(x * x + y * y)

    # Initializing sigma and muu
    sigma_ = 0.1
    mu = 0.000

    # Calculating Gaussian array
    gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma_ ** 2)))
    x = torch.randn((1,1,128,128))*sigma_max
    lambdas = torch.from_numpy(gauss).unsqueeze(0).unsqueeze(0).float().to('cpu')
    obs_f = reorganize(FT.fftn(obs, s=None, dim=(-2, -1), norm=None))
    with tqdm.tqdm(total=steps, desc=f'Reverse SDE RK') as pbarv:
        for i in reversed(range(1,steps)):
            # masking noise
            x = x.to('cpu')
            obs_f_t = mask*obs_f + sigma[i]*mask*reorganize(FT.fftn(torch.randn(1,1,128,128), s=None, dim=(-2, -1), norm=None))
            x_f = FT.fftn(x, s=None, dim=(-2, -1), norm=None)
            x_f = reorganize(x_f)
            x_f_mask = x_f * mask
            x_f_mask_c = x_f * mask_complement
            x_kspace = (1-lambdas)*x_f_mask + x_f_mask_c + lambdas*obs_f_t

            x_kspace = reorganize(x_kspace)
            x_real = torch.real(FT.ifftn(x_kspace, s=None, dim=(-2, -1), norm=None))
            ###########################################################################
            
            # Calculate score
            score = score_model(x_real.to(device), dt[i].to(device)).to('cpu')  

            # Predictor Step REAL
            x = x_real.to('cpu')
            g = (sigma[i] ** 2 - sigma[i - 1] ** 2)
            g_sqrt = torch.sqrt(g)
            g2 = (sigma[i - 1] ** 2 - sigma[i - 2] ** 2)
            g2_sqrt = torch.sqrt(g2)
        
            x_temp = x + score*g + torch.randn(1,1,128,128)*g_sqrt
            score_2 = score_model(x_temp.to(device), dt[i-1].to(device)).to('cpu')
            x_hat = x + 0.5*torch.randn(1,1,128,128)*g_sqrt+0.5*torch.randn(1,1,128,128)*g2_sqrt
            x = x_hat + 0.5*score*g + 0.5*score_2*g2

            # Corrector Step REAL
            z = torch.randn(x.shape).to('cpu')
            z_norm = torch.sqrt(torch.sum(z[:, 0, :, :] ** 2))
            s_norm = torch.sqrt(torch.sum(score[:, 0, :, :] ** 2))
            epsilon = 2 * ((r * z_norm / s_norm) ** 2)
            x_mean_real = x + epsilon * score
            x = x_mean_real + torch.sqrt(2 * epsilon) * z

            if i>1000:
                if i%250==0:
                    sde_img = x_mean_real.to('cpu')
                    plt.figure()
                    plt.imshow(sde_img[0,0,:,:])
                    plt.savefig(f'{cnt}gif')
                    cnt +=1

            if i<1000 and i>100:
                if i%100==0:
                    sde_img = x_mean_real.to('cpu')
                    plt.figure()
                    plt.imshow(sde_img[0,0,:,:])
                    plt.savefig(f'{cnt}gif')
                    cnt +=1

            if i<100 and i>10:
                if i%10==0:
                    sde_img = x_mean_real.to('cpu')
                    plt.figure()
                    plt.imshow(sde_img[0,0,:,:])
                    plt.savefig(f'{cnt}gif')
                    cnt +=1

            if i<10:
                sde_img = x_mean_real.to('cpu')
                plt.figure()
                plt.imshow(sde_img[0,0,:,:])
                plt.savefig(f'{cnt}gif')
                cnt +=1

            pbarv.update(1)

    sde_img = x_mean_real.to('cpu')

plt.figure()
plt.imshow(tar[0,0,:,:])
plt.savefig('GNDgif')