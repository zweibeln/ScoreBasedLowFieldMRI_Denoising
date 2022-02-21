from imports import *


def dc_in_mnist(x, observation, target, model, optimizer, Loss, device='cuda'if torch.cuda.is_available() else 'cpu'):
    optimizer.zero_grad()
    x = x.to(device)
    x = x + model(observation) - model(x)
    loss = Loss(x, target)
    loss.backward()
    optimizer.step()
    return x
