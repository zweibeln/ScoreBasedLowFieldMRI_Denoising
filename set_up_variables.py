from configs import *


# SET UP SET TYPE
set_type = config.set_type  # mnist, mri

# SET UP DATA CONSISTENCY
mri_dc = config.mri.dc
mnist_dc = config.mnist.dc
if config.mnist.dc:
    data_consistency = 'mnist'
elif config.mri.dc:
    data_consistency = 'mri'
else:
    data_consistency = False

# SET UP MODEL
model_name = config.model.model_name  # unet_wot
out_sz = config.mri.out_sz if set_type == 'mri' else config.mnist.out_sz  # corresponding output_size given set_type
retain_dim = config.model.retain_dim  # if out_sz of network do not match the given one then it interpolates
initialize_weights = config.model.initialize_weights  # initialize weights as kaiming normal activation relu
num_class = config.model.num_class  # number of output class
enc_chs = config.model.enc_chs  # number of channels in encoder block (1,64,128,256,512,1024)
dec_chs = config.model.dec_chs  # number of decoder channels in decoder block (1024,512,256,128,64)

# SET UP TRAINING
device = config.training.device  # device = cuda if available
adam_lr = config.training.lr.adam  # learning rate for adam optimizer
betas = config.training.betas  # beta values for adam optimizer
sgd_lr = config.training.lr.sgd  # learning rate for sgd optimizer
milestones = config.training.milestones  # learning rate scheduler mile stones to reduce learning rate
gamma = config.training.gamma  # Multiplicative factor of reduction in learning rate
ema_value = config.training.ema  # Exponential moving average
num_epochs = config.training.epochs  # number of epochs for training
grad_clip_norm = config.training.grad_clip_norm  # gradient clipping (=1 default)
field_str_score = config.mri.field_str_score  # choosing field strength 1.5T for training score model in mri
centralize_data = config.training.centralize_data  # make data dist ~ N(0,1)

# SET UP SAMPLING
t_0 = config.sampling.t_0  # SDE 0th time
Tmax = config.sampling.T  # Max SDE time (=1)
steps = config.sampling.steps  # Number of predictor steps to solve SDE
M = config.sampling.M  # Number of corrector steps to solve SDE
r = config.sampling.snr  # SNR value
dt = torch.linspace(t_0, Tmax, steps=steps).to(device)  # Time bins
img_nums = config.mnist.img if config.set_type == 'mnist' else config.mri.img  # Chosen images to be reconstructed
sigma_max = config.mnist.sigma_max if config.set_type == 'mnist' else config.mri.sigma_max  # Maximum noise std
sigma_min = config.mnist.sigma_min if config.set_type == 'mnist' else config.mri.sigma_min  # Min noise std
sigma = (sigma_min*((sigma_max/sigma_min)**dt)).float().to('cpu')  # Corresponding sigmas throughout process
langevin = config.sampling.langevin  # Boolean (default=False), determines if corrector steps performed
dc_loss = config.sampling.dc_loss.ssim  # Data consistency Loss
dc_lr = config.sampling.dc_lr  # Data consistency learning rate
dc_f = config.sampling.dc_f  # MRI data field strength for data consistency stage
dc_img_patch = config.sampling.dc_img_patch  # Image patch 











