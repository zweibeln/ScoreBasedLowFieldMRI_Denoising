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

def PSNR(image, target):
    mse = torch.mean((image-target)**2)
    max_i = torch.max(image)
    PSNR_val = 20*torch.log10(max_i) - 10*torch.log10(mse)
    return PSNR_val


# LOADING TRAINED MODELS
score_model, _,_,_,_ = init_model()

# CREATE DCTIONARY
field_str = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
sde_name = ['ccdf_ssim','ccdf_psnr','ocmccdf_ssim','ocmccdf_psnr']  # 0,1,2,3
sde_stats = {}
for f in field_str:
    sde_stats[str(f)] = {}

for f in field_str:
    for m in sde_name:
        sde_stats[str(f)][m] = {}

ssim = SSIM_loss(win_sz=4)

# LOADING DATA SET
for f_str in field_str:
    ccdf_ssim = []
    ccdf_psnr = []
    ocmccdf_ssim = []
    ocmccdf_psnr = []
    randomseed=97
    set_type = 'val'
    obs_, mask_, target_, noise_ = draw_data(f_str,randomseed=randomseed, set_type=set_type, num_volumes=50, return_std=1)
    if f_str<0.7:
        num_of_unrolls = 4
    if f_str<0.9 and f_str>0.6:
        num_of_unrolls = 3
    if f_str<1.1 and f_str>0.8:
        num_of_unrolls = 2
    if f_str<1.2 and f_str>1.0:
        num_of_unrolls = 1
    if f_str>1.1:
        num_of_unrolls = 0

    for idx,(obs,mask,target,noise) in enumerate(zip(obs_,mask_,target_,noise_)):
        print(f'CURRENT FIELD STR = {f_str}')
        ########### STANDARDIZING THE IMAGES ###############
        obs = obs - torch.min(obs)
        obs = obs/torch.max(obs)
        tar = target - torch.min(target)
        tar = tar/torch.max(tar)
        torch.set_grad_enabled(False)
        mask_complement = 1-mask
        obs_f = reorganize(FT.fftn(obs, s=None, dim=(-2, -1), norm=None))

        ## CCDF
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha) if int(steps * t * alpha)>0 else 90
        lambdas = 0.0056
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type=None)
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_ccdf = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        psnr_ccdf = PSNR(rec_imp.to('cpu'), tar)
        ccdf_ssim.append(rec_ssim_ccdf)
        ccdf_psnr.append(psnr_ccdf)

        ## OCMCCDF
        noise_level = noise_level_estimation(obs)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        alpha = 0.2
        steps_denoising = int(steps * t * alpha) if int(steps * t * alpha)>0 else 90
        lambdas = 0.0056
        sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type='mean')
        rec_imp = sde_img - torch.min(sde_img)
        rec_imp = rec_imp/torch.max(rec_imp)
        rec_ssim_ocmccdf = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
        noise_level = noise_level_estimation(rec_imp)
        for i in range(num_of_unrolls-1):
            t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
            steps_denoising = int(steps * t * alpha) if int(steps * t * alpha)>0 else int(90-i*5)
            sde_img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=sde_img, sample_type='mean')
            rec_imp = sde_img - torch.min(sde_img)
            rec_imp = rec_imp/torch.max(rec_imp)
            rec_ssim_ocmccdf = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
            noise_level = noise_level_estimation(rec_imp)
        psnr_ocmccdf = PSNR(rec_imp.to('cpu'), tar)
        ocmccdf_ssim.append(rec_ssim_ocmccdf)
        ocmccdf_psnr.append(psnr_ocmccdf)

    sde_stats[str(f_str)]['ccdf_ssim'] = ccdf_ssim
    sde_stats[str(f_str)]['ccdf_psnr'] = ccdf_psnr
    sde_stats[str(f_str)]['ocmccdf_ssim'] = ocmccdf_ssim
    sde_stats[str(f_str)]['ocmccdf_psnr'] = ocmccdf_psnr
    torch.save(sde_stats, 'ccdf_ocmccdf.txt')

torch.save(sde_stats, 'ccdf_ocmccdf.txt')