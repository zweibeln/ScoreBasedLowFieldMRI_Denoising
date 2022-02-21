from imports import *
from produce_data import *
from dc_stepsize import *
from dc_in_mri import *
from init_model import *
from set_up_variables import *
from predictor import *
from corrector import *
from mri_data_consistency import *
from mnist_data_consistency import *
from plot_images import *
from arrange_data_for_mnist_sampling import *
from arrange_data_for_mri_sampling import *
from print_configs import *

print_configs()

if mri_dc:
    observation, mask, target = arrange_data_for_mri_sampling()
if mnist_dc:
    observation, target = arrange_data_for_mnist_sampling()

score_model,_,_,_,_ = init_model()
x = torch.randn((4,1,128,128))*sigma_max if set_type=='mri' else torch.randn((64,1,32,32))*sigma_max

if mnist_dc or mri_dc:
    # Construct DC_Stepsize
    dc_stepsize = DC_StepSize().to(device)
    Loss = dc_loss[0]  # SSIM_loss(), nn.MSELoss()
    optimizer = optim.Adam(dc_stepsize.parameters(), lr=dc_lr, betas=(0.9, 0.999))
    print(f'CHOSEN LOSS OF DC {dc_loss[1]}')
    x = observation.repeat(dc_img_patch,1,1,1)

torch.set_grad_enabled(False)
with tqdm.tqdm(total=steps, desc=f'Reverse SDE') as pbarv:
    for i in reversed(range(1,steps)):
        # Predictor Step
        x = predictor(x, score_model, i)
        # Data Consistency
        if mri_dc:
            torch.set_grad_enabled(True)
            x = dc_mri(x, mask, observation, target, dc_stepsize, optimizer, Loss, device='cuda' if torch.cuda.is_available() else 'cpu')
            torch.set_grad_enabled(False)
        if mnist_dc:
            torch.set_grad_enabled(True)
            x = dc_in_mnist(x, observation, target, dc_stepsize, optimizer, Loss, device='cuda' if torch.cuda.is_available() else 'cpu')
            torch.set_grad_enabled(False)
        if langevin:
            for j in range(M):
                # Corrector Step
                x = corrector(x, score_model, i)
                if mri_dc:
                    torch.set_grad_enabled(True)
                    x = dc_mri(x, mask, observation, target, dc_stepsize, optimizer, Loss, device='cuda' if torch.cuda.is_available() else 'cpu')
                    torch.set_grad_enabled(False)
                if mnist_dc:
                    torch.set_grad_enabled(True)
                    x = dc_in_mnist(x, observation, target, dc_stepsize, optimizer, Loss, device='cuda' if torch.cuda.is_available() else 'cpu')
                    torch.set_grad_enabled(False)

        pbarv.update(1)

x = x.to('cpu')

# Plots
plot_images(x, f'RECONSTRUCTED_{dc_loss[1]}')
if data_consistency:
    plot_images(target, 'GND')
    plot_images(observation, 'NOISY')

    # Calculate SSIM and SNR
    ssim = SSIM_loss()
    nois_ssim = 1 - 2 * ssim(observation, target)
    rec_ssim = 1 - 2 * ssim(x, target)
    print(f'SSIM BEFORE {model_name} = {nois_ssim}')
    print(f'SSIM AFTER {model_name} = {rec_ssim}')

    snr_before, _ = snr_calculate(observation[0, 0, :, :])
    snr_after, _ = snr_calculate(x[0, 0, :, :])
    print(f'SNR BEFORE {model_name} = {snr_before}')
    print(f'SNR AFTER {model_name} = {snr_after}')

