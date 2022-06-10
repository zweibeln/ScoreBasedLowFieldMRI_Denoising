from configs.imports import * 
from configs.set_up_variables import *
from models.init_model import *
from configs.print_configs import *
from losses.ssim_loss import *
from utils.draw_data import *
import torchvision.transforms.functional as TF
from numpy import linalg as LA


def extract_patches(obs, patch_size):
    shp = obs.shape
    number_of_total_patches = (shp[-2]-patch_size+1)*(shp[-1]-patch_size+1)
   
    patches = []
    for i in range(shp[-2]-patch_size):
        for j in range(shp[-1]-patch_size):
            patches.append(np.expand_dims(obs[0,0,i:i+patch_size,j:j+patch_size].detach().numpy().reshape(-1), axis=1))

    return patches

def eig(covariance):
    eig_vals = LA.eigvals(covariance)
    eig_vals[::-1].sort()
    return eig_vals


def noise_level_estimation(obs):
    '''
        obs -> torch.tensor: (1,1,128,128) 
    '''
    patch_size = 4  # 4x4=16 dim vectors
    patches = extract_patches(obs, patch_size)  # 16x1 vectors

    mean = 0
    for p in patches:
        mean = mean + p
    mean = mean/len(patches)

    covariance = 0
    for p in patches:
        covariance = covariance + (p-mean)*(p-mean).T
    covariance = covariance/len(patches)

    eigen_vals = eig(covariance)

    for i in range(int(patch_size**2)):
        tao = np.sum(eigen_vals[i:])*(1/len(eigen_vals[i:]))
        median = np.median(eigen_vals[i:])
        if tao == median:
            sigma = np.sqrt(tao)
            break
    
    # print(f'sigma = {sigma}')
    return sigma

def get_median_img(recon):
    '''
    recon : (M,1,128,128) tensor
    '''
    temp = torch.empty(1,1,recon.shape[-2],recon.shape[-1])
    for i in range(recon.shape[-2]):
        for j in range(recon.shape[-1]):
            temp[0,0,i,j] = torch.quantile(recon[:,0,i,j], q=0.5)
    
    return temp
    

def DataConsistency(x, obs_f, mask, mask_complement, sigma, i, lambdas):
    x = x.to('cpu')
    obs_f_t = mask*obs_f + sigma[i]*mask*reorganize(FT.fftn(torch.randn(1,1,128,128), s=None, dim=(-2, -1), norm=None))
    x_f = FT.fftn(x, s=None, dim=(-2, -1), norm=None)
    x_f = reorganize(x_f)
    x_f_mask = x_f * mask
    x_f_mask_c = x_f * mask_complement
    x_kspace = (1-lambdas)*x_f_mask + x_f_mask_c + lambdas*obs_f_t

    x_kspace = reorganize(x_kspace)
    return torch.real(FT.ifftn(x_kspace, s=None, dim=(-2, -1), norm=None)).to(device)

def Predictor(x, score, sigma, i):
    # Predictor Step REAL
    x = x.to('cpu')
    x_mean = x + (sigma[i] ** 2 - sigma[i - 1] ** 2) * score
    z = torch.randn(x.shape).to('cpu')
    return x_mean + torch.sqrt(sigma[i] ** 2 - sigma[i - 1] ** 2) * z

def Corrector(x, score, sigma, i, r):
    z = torch.randn(x.shape).to('cpu')
    z_norm = torch.sqrt(torch.sum(z[:, 0, :, :] ** 2))
    s_norm = torch.sqrt(torch.sum(score[:, 0, :, :] ** 2))
    epsilon = 2 * ((r * z_norm / s_norm) ** 2)
    x_mean = x + epsilon * score
    x = x_mean + torch.sqrt(2 * epsilon) * z
    if i==1:
        return x_mean
    else:
        return x

def GetLambda(sigma_, scale, lift):
    x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    dst = np.sqrt(x * x + y * y)

    # Initializing sigma and muu
    mu = 0.000

    # Calculating Gaussian array
    gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma_ ** 2)))

    gauss = gauss/gauss.max()
    gauss = gauss*scale+lift
    return torch.from_numpy(gauss).unsqueeze(0).unsqueeze(0).float().to('cpu')

def PC_Denoising_Sampling(steps, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=None, sample_type='mean', median=False):
    with torch.no_grad(), tqdm.tqdm(total=steps-1, desc=f'Reverse SDE IMP') as pbar:
        if sample_type == 'mean':
            x = torch.randn((16,1,128,128))*sigma_max if initial==None else initial.repeat(16,1,1,1)
        else:
            x = torch.randn((1,1,128,128))*sigma_max if initial==None else initial.repeat(1,1,1,1)
        for i in reversed(range(1,steps)):
            x = DataConsistency(x, obs_f, mask, mask_complement, sigma, i, lambdas)
            score = score_model(x,dt[i]).to('cpu')
            x = Predictor(x, score, sigma, i)
            x = Corrector(x, score, sigma, i, r)
            pbar.update(1)
    x = x.to('cpu')
    if median:
        x = get_median_img(x)
    else:
        x = torch.sum(x, dim=0)/x.shape[0]
        x = x.unsqueeze(0)

    return x


# LOADING TRAINED MODELS
score_model, _,_,_,_ = init_model()

# CREATE DCTIONARY
f_str = [1.45, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
sde_name = ['gauss_ssim','gauss_noise','cascaded_sde_ssim','cascaded_sde_noise']  # 0,1,2,3
sde_stats = {}
for f in f_str:
    sde_stats[str(f)] = {}

for f in f_str:
    for m in sde_name:
        sde_stats[str(f)][m] = {}

ssim = SSIM_loss(win_sz=4)

# LOADING DATA SET
# observation, mask_, target_ = produce_data_sampling2(field_str_tr=[f_str,f_str], resizing=(128,128), set='train')
for f in f_str:
    print(f'CURRENT FIELD STR = {f}')
    randomseed=101
    set_type = 'val'
    obs_, mask_, target_, noise_ = draw_data(f,randomseed=randomseed, set_type=set_type, num_volumes=25, return_std=1)
    gauss_sde = []
    gauss_noise = []
    cas_sde = []
    cas_noise = []

    for idx,(obs,mask,target,noise) in enumerate(zip(obs_,mask_,target_,noise_)):
        ########### STANDARDIZING THE IMAGES ###############
        obs = obs - torch.min(obs)
        obs = obs/torch.max(obs)
        tar = target - torch.min(target)
        tar = tar/torch.max(tar)
        torch.set_grad_enabled(False)
        mask_complement = 1-mask
        obs_f = reorganize(FT.fftn(obs, s=None, dim=(-2, -1), norm=None))

        # INFERENCE
        ## GausLambda SDE
        print(f'GAUSS SDE MEAN SOLUTION F_STR = {f}')
        lambdas = GetLambda(0.005743640564259807, 0.6513385788564359, 0.004648959216765907)
        cascaded_sde_noise = []
        cascaded_sde_ssim = []
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha)
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type='mean')
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        cascaded_sde_ssim.append(rec_ssim_imp)
        cascaded_sde_noise.append(noise_level)
        for i in range(9):
            t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
            steps_denoising = int(steps * t * alpha) if int(steps * t * alpha)>0 else int(90-i*5)
            sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=sde_img, sample_type='mean')
            rec_imp = sde_img - torch.min(sde_img)
            rec_imp = rec_imp/torch.max(rec_imp)
            rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
            noise_level = noise_level_estimation(rec_imp)
            cascaded_sde_ssim.append(rec_ssim_imp)
            cascaded_sde_noise.append(noise_level)
        
        gauss_sde.append(cascaded_sde_ssim)
        gauss_noise.append(cascaded_sde_noise)

        ## CASCADED SDE MEAN
        cascaded_sde_noise = []
        cascaded_sde_ssim = []
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha)
        lambdas = 0.0056
        print(f'CASCADED SDE MEAN SOLUTION F_STR = {f}')
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type='mean')
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        cascaded_sde_ssim.append(rec_ssim_imp)
        cascaded_sde_noise.append(noise_level)
        for i in range(9):
            t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
            steps_denoising = int(steps * t * alpha) if int(steps * t * alpha)>0 else int(90-i*5)
            sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=sde_img, sample_type='mean')
            rec_imp = sde_img - torch.min(sde_img)
            rec_imp = rec_imp/torch.max(rec_imp)
            rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
            noise_level = noise_level_estimation(rec_imp)
            cascaded_sde_ssim.append(rec_ssim_imp)
            cascaded_sde_noise.append(noise_level)
        
        cas_sde.append(cascaded_sde_ssim)
        cas_noise.append(cascaded_sde_noise)

    sde_stats[str(f)][sde_name[0]] = gauss_sde  # normal_sde
    sde_stats[str(f)][sde_name[1]] = gauss_noise  
    sde_stats[str(f)][sde_name[2]] = cas_sde  # cascaded mean
    sde_stats[str(f)][sde_name[3]] = cas_noise

torch.save(sde_stats,'cascaded_vs_gauss_stats.txt')


    






