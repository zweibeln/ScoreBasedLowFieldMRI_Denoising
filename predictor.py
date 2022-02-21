from set_up_variables import *


def predictor(x, score_model, i):
    x = x.to(device)
    torch.set_grad_enabled(False)
    score = score_model(x, dt[i])
    score = score.to('cpu')
    x = x.to('cpu')
    x = x + (sigma[i] ** 2 - sigma[i - 1] ** 2) * score
    if i > 1:
        z = torch.randn(x.shape).to('cpu')
    else:
        z = 0
    x = x + torch.sqrt(sigma[i] ** 2 - sigma[i - 1] ** 2) * z
    return x