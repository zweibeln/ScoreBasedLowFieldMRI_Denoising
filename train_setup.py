from mnist_ import *
from produce_data import *
from fit_model import *


if set_type == 'mnist':
    print(f'Chosen Set Type Is {set_type}')
    x_train_mnist = load_mnist()
    canvas = torch.ones((int(x_train_mnist.shape[0]), int(x_train_mnist.shape[1]), 32, 32))*torch.min(x_train_mnist)
    canvas[:, :, 2:30, 2:30] = x_train_mnist
    x_train = [x_[None, ...].to(device) for x_ in canvas]

if set_type == 'mri':
    print(f'Chosen Set Type Is {set_type}')
    # Load data to train score model
    x_train = produce_data_train()
    x_train = [x.to(device) for x in x_train]

total_loss_tr = fit_model(x_train)

plt.figure()
plt.plot(total_loss_tr, 'b-', label='ScoreFunctionLoss_Train')
plt.legend()
plt.title('Score Loss')
plt.xlabel('epoch')
plt.savefig(f'ScoreLoss_{model_name}_{set_type}.png')
