from imports import *
from dc_in_mri import *
from set_up_variables import dc_img_patch


def dc_mri(x, mask, observation, target, model, optimizer, Loss, device='cuda' if torch.cuda.is_available() else 'cpu'):
    x = x.to('cpu')
    mask = mask.to('cpu')
    x, x_masked = dc_in_mri(x, mask, device='cpu')
    optimizer.zero_grad()
    x = x.to(device)
    x_masked = x_masked.to(device)
    x = x + model(observation) - model(x_masked)
    loss = Loss(x, target.repeat(dc_img_patch, 1, 1, 1))
    loss.backward()
    optimizer.step()
    return x
