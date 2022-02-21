from imports import *
from set_up_variables import *


def time_std(t, sigma_min=sigma_min, sigma_max=sigma_max):
    '''
    inputs:
        t : torch.tensor -> dims=(1,1), time ~ U[0,1]
        sigma_min : float
        sigma_max : float
    outputs:
        sigma_t : torch.tensor(1), time varying std
    '''
    return (sigma_min*((sigma_max/sigma_min)**t)).float() # .to(device)


def score_loss(model, x, device, sigma_min=sigma_min, sigma_max=sigma_max):
    '''
    inputs:
        model: nn.Module -> Time Dependent Score Based Model
        x : torch.tensor -> (B,C,H,W)
    outputs:
        loss : torch.tensor
    '''
    t = torch.unsqueeze(torch.from_numpy(np.random.uniform(1e-5, 1, 1)), 0).float().to(device)  # randomly selected t
    std_t = time_std(t, sigma_max=sigma_max).to(device)  # time varying std
    z = torch.randn(x.shape).to(device)  # Gaussian random noise
    xt = x + std_t * z  # xt = x0 + sigma_t*N(0,1)
    # time dependent score model loss assuming Lambda(t)=sigma^2
    loss = torch.mean(torch.sum((model(xt, t)*std_t+z)**2, dim=(-2, -1)))
    return loss
