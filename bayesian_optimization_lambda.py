from configs.imports import * 
from configs.set_up_variables import *
from models.init_model import *
from configs.print_configs import *
from losses.ssim_loss import *
from utils.draw_data import *
import torchvision.transforms.functional as TF
from numpy import linalg as LA

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from numpy import linalg as LA
# import lightgbm as lgb

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
    with torch.no_grad():#, tqdm.tqdm(total=steps-1, desc=f'Reverse SDE IMP') as pbar:
        if sample_type == 'mean':
            x = torch.randn((16,1,128,128))*sigma_max if initial==None else initial.repeat(16,1,1,1)
        else:
            x = torch.randn((1,1,128,128))*sigma_max if initial==None else initial.repeat(1,1,1,1)
        for i in reversed(range(1,steps)):
            x = DataConsistency(x, obs_f, mask, mask_complement, sigma, i, lambdas)
            score = score_model(x,dt[i]).to('cpu')
            x = Predictor(x, score, sigma, i)
            x = Corrector(x, score, sigma, i, r)
            # pbar.update(1)
    x = x.to('cpu')
    if median:
        x = get_median_img(x)
    else:
        x = torch.sum(x, dim=0)/x.shape[0]
        x = x.unsqueeze(0)

    return x

def cascaded_sde(steps, dt, sigma, obs, obs_f, mask, mask_complement, r, initial=None, sample_type='mean', median=False, sigma_=0, scale=0, lift=0):
    alpha = 0.2
    lambdas = GetLambda(sigma_, scale, lift)
    noise_level = noise_level_estimation(obs)
    t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
    steps_denoising = int(steps * t * alpha) if int(steps * t * alpha)>0 else int(90)
    img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=obs, sample_type='mean', median=False)
    rec_img = img - torch.min(img)
    rec_img = rec_img/torch.max(rec_img)
    for i in range(9):
        noise_level = noise_level_estimation(rec_img)
        t = np.log(noise_level/sigma_min)/np.log(sigma_max/sigma_min)
        steps_denoising = int(steps * t * alpha) if int(steps * t * alpha)>0 else int(90-i*5)
        img = PC_Denoising_Sampling(steps_denoising, dt, sigma, lambdas, obs_f, mask, mask_complement, r, initial=img, sample_type='mean', median=False)
        rec_img = img - torch.min(img)
        rec_img = rec_img/torch.max(rec_img)

    return img



def objective_function(params): #, steps, dt, sigma, obs_f, mask, mask_complement, r):
    ssim = SSIM_loss(win_sz=4)
    sigma_, scale, lift = params['std_dev'],params['scale'],params['lift']
    sde_img_imp = cascaded_sde(params['steps'], params['dt'], params['sigma'], params['obs'], params['obs_f'], params['mask'], params['mask_complement'], params['r'],
    params['initial'], params['sample_type'], params['median'], sigma_, scale, lift)
    ################################# CHECK SSIM ##########################################################
    rec_imp = sde_img_imp - torch.min(sde_img_imp)
    rec_imp = rec_imp/torch.max(rec_imp)
    rec_ssim_imp = 1 - 2 * ssim(rec_imp.to('cpu'), tar)
    print(f'SSIM = {rec_ssim_imp}')
    score = rec_ssim_imp
    return {'loss': -score, 'status': STATUS_OK}


# Init Model
score_model, _,_,_,_ = init_model()
# LOADING DATA SET
f_str = 0.5
# observation, mask_, target_ = produce_data_sampling2(field_str_tr=[f_str,f_str], resizing=(128,128), set='train')
randomseed=56
set_type = 'val'
obs_, mask_, target_ = draw_data(f_str,randomseed=randomseed, set_type=set_type, num_volumes=1)

for idx,(obs,mask,target) in enumerate(zip(obs_,mask_,target_)):

    ########### STANDARDIZING THE IMAGES ###############
    obs = obs - torch.min(obs)
    obs = obs/torch.max(obs)
    tar = target - torch.min(target)
    tar = tar/torch.max(tar)
    obs_f = reorganize(FT.fftn(obs, s=None, dim=(-2, -1), norm=None))

    # ############################## Improved Lambdas SDE ##################################################
    mask_complement = 1-mask

param_hyperopt= {
    'std_dev': hp.normal('std_dev', 0.0556, 0.01),#hp.uniform('std_dev', 0.0001, 0.1), -> 0.0556
    'scale': hp.normal('scale', 0.831, 0.05),#hp.normal('scale', 0.5, 0.1), -> 0.831
    'lift': hp.normal('lift', 0.0014, 0.001),#hp.uniform('lift', 0., 0.5), -> 0.0014
    'steps': steps,
    'dt': dt,
    'sigma': sigma,
    'obs': obs,
    'obs_f': obs_f,
    'mask': mask,
    'mask_complement': mask_complement,
    'r': r,
    'initial': None,
    'sample_type': 'mean',
    'median': False
}


# Initialize trials object
trials = Trials()

best = fmin(
    fn=objective_function,
    space = param_hyperopt, 
    algo=tpe.suggest, 
    max_evals=125, 
    trials=trials
)

print("Best: {}".format(best))

#Best: {'lift': 0.004648959216765907, 'scale': 0.6513385788564359, 'std_dev': 0.005743640564259807}
