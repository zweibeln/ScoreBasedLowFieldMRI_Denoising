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
            score = score_model(x.to(device),dt[i-1]).to('cpu')
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
sde_name = ['trunc_sde_ssim','trunc_sde_noise','cascaded_sde_ssim','cascaded_sde_noise', # 0,1,2,3
'trunc_sde_mean_ssim','trunc_sde_mean_noise','cascaded_sde_mean_ssim','cascaded_sde_mean_noise', # 4,5,6,7
'trunc_sde_median_ssim','trunc_sde_median_noise','cascaded_sde_median_ssim','cascaded_sde_median_noise'] # 8,9,10,11
sde_stats = {}
for f in f_str:
    sde_stats[str(f)] = {}

for f in f_str:
    for m in sde_name:
        sde_stats[str(f)][m] = {}

ssim = SSIM_loss(win_sz=4)

# LOADING DATA SET
for f in f_str:
    print(f'CURRENT FIELD STR = {f}')
    randomseed=101
    set_type = 'val'
    obs_, mask_, target_, noise_ = draw_data(f,randomseed=randomseed, set_type=set_type, num_volumes=25, return_std=1)
    ss_sde = []
    ss_noise = []
    cas_sde = []
    cas_noise = []
    ss_mean = []
    ss_mean_n = []
    cas_mean = []
    cas_mean_n = []
    ss_median = []
    ss_median_n = []
    cas_median = []
    cas_median_n = []

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
        ## CCDF
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha)
        lambdas = 0.0056
        print(f'TRUNC SDE SOLUTION F_STR = {f}')
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type=None)
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        ss_sde.append(rec_ssim_imp)
        ss_noise.append(noise_level)
        
        ## CASCADED CCDF
        cascaded_sde_noise = []
        cascaded_sde_ssim = []
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha)
        lambdas = 0.0056
        print(f'CASCADED SDE SOLUTION F_STR = {f}')
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type=None)
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        cascaded_sde_ssim.append(rec_ssim_imp)
        cascaded_sde_noise.append(noise_level)
        for i in range(9):
            t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
            steps_denoising = int(steps * t * alpha) if steps_denoising>0 else int(90-i*5)
            sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=sde_img, sample_type=None)
            rec_imp = sde_img - torch.min(sde_img)
            rec_imp = rec_imp/torch.max(rec_imp)
            rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
            noise_level = noise_level_estimation(rec_imp)
            cascaded_sde_ssim.append(rec_ssim_imp)
            cascaded_sde_noise.append(noise_level)
        
        cas_sde.append(cascaded_sde_ssim)
        cas_noise.append(cascaded_sde_noise)

        ## MEAN CCDF
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha)
        lambdas = 0.0056
        print(f'TRUNC SDE MEAN SOLUTION F_STR = {f}')
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type='mean')
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        ss_mean.append(rec_ssim_imp)
        ss_mean_n.append(noise_level)

        ## CASCADED MEAN CCDF
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
            steps_denoising = int(steps * t * alpha) if steps_denoising>0 else int(90-i*5)
            sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=sde_img, sample_type='mean')
            rec_imp = sde_img - torch.min(sde_img)
            rec_imp = rec_imp/torch.max(rec_imp)
            rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
            noise_level = noise_level_estimation(rec_imp)
            cascaded_sde_ssim.append(rec_ssim_imp)
            cascaded_sde_noise.append(noise_level)
        
        cas_mean.append(cascaded_sde_ssim)
        cas_mean_n.append(cascaded_sde_noise)
        
        # MEDIAN CCDF
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha)
        lambdas = 0.0056
        print(f'TRUNC SDE MEDIAN SOLUTION F_STR = {f}')
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type='mean', median=True)
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        ss_median.append(rec_ssim_imp)
        ss_median_n.append(noise_level)
        
        ## CASCADED MEDIAN CCDF
        cascaded_sde_noise = []
        cascaded_sde_ssim = []
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha)
        lambdas = 0.0056
        print(f'CASCADED SDE MEDIAN SOLUTION F_STR = {f}')
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type='mean', median=True)
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        cascaded_sde_ssim.append(rec_ssim_imp)
        cascaded_sde_noise.append(noise_level)
        for i in range(9):
            t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
            steps_denoising = int(steps * t * alpha) if steps_denoising>0 else int(90-i*5)
            sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=sde_img, sample_type='mean', median=True)
            rec_imp = sde_img - torch.min(sde_img)
            rec_imp = rec_imp/torch.max(rec_imp)
            rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
            noise_level = noise_level_estimation(rec_imp)
            cascaded_sde_ssim.append(rec_ssim_imp)
            cascaded_sde_noise.append(noise_level)
        
        cas_median.append(cascaded_sde_ssim)
        cas_median_n.append(cascaded_sde_noise)

    sde_stats[str(f)][sde_name[0]] = ss_sde  # CCDF
    sde_stats[str(f)][sde_name[1]] = ss_noise  
    sde_stats[str(f)][sde_name[2]] = cas_sde  # cascaded CCDF
    sde_stats[str(f)][sde_name[3]] = cas_noise
    sde_stats[str(f)][sde_name[4]] = ss_mean  # MEAN CCDF
    sde_stats[str(f)][sde_name[5]] = ss_mean_n
    sde_stats[str(f)][sde_name[6]] = cas_mean  # CASCADED MEAN CCDF
    sde_stats[str(f)][sde_name[7]] = cas_mean_n
    sde_stats[str(f)][sde_name[8]] = ss_median  # MEDIA CCDF
    sde_stats[str(f)][sde_name[9]] = ss_median_n
    sde_stats[str(f)][sde_name[10]] = cas_median  # Cascaded Median CCDF
    sde_stats[str(f)][sde_name[11]] = cas_median_n


torch.save(sde_stats,'sde_stats.txt')


    






