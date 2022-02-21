from set_up_variables import *


def corrector(x, score_model, i):
    # Corrector
    z = torch.randn(x.shape).to('cpu')
    z_norm = torch.sqrt(torch.sum(z[:, 0, :, :] ** 2))
    x = x.to(device)
    score = score_model(x, dt[i]).to(device)
    score = score.to('cpu')
    x = x.to('cpu')
    s_norm = torch.sqrt(torch.sum(score[:, 0, :, :] ** 2))
    epsilon = 2 * r * z_norm / s_norm
    x = x + epsilon * score + torch.sqrt(2 * epsilon) * z
    return x
