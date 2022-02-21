from set_up_variables import *
from mnist_ import *
from mpl_toolkits.axes_grid1 import ImageGrid


def arrange_data_for_mnist_sampling():
    print(f'Data Consistency: {set_type}')
    x_train_mnist = load_mnist()
    canvas = torch.ones((int(x_train_mnist.shape[0]), int(x_train_mnist.shape[1]), 32, 32)) * torch.min(x_train_mnist)
    canvas[:, :, 2:30, 2:30] = x_train_mnist
    target = torch.zeros((64, 1, 32, 32)).to(device)
    x_train = [x_[None, ...].to(device) + torch.randn((1, 1, 32, 32)).to(device) for x_ in canvas]
    img = torch.zeros((64, 1, 32, 32)).to(device)
    for idx in img_nums:
        img[idx, :, :, :] = x_train[idx][0, :, :, :]
        target[idx, :, :, :] = canvas[idx, :, :, :]

    fig = plt.figure(figsize=(20., 20.))
    plt.style.use('grayscale')
    plt.title(f'MNIST RECONS')
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(8, 8),
                     axes_pad=0.,  # pad between axes in inch.
                     )
    for idx, ax in enumerate(grid):
        ax.imshow(target[idx, 0, :, :].to('cpu').detach().numpy())
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'MNIST TARGET DC={mnist_dc}.png')
    print(f'SAVED FIGURE AS MNIST TARGET={mnist_dc}')

    fig = plt.figure(figsize=(20., 20.))
    plt.style.use('grayscale')
    plt.title(f'MNIST RECONS')
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(8, 8),
                     axes_pad=0.,  # pad between axes in inch.
                     )
    for idx, ax in enumerate(grid):
        ax.imshow(img[idx, 0, :, :].to('cpu').detach().numpy())
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(f'MNIST NOISY DC={mnist_dc}_N={steps}_M={M}.png')
    print(f'SAVED FIGURE AS MNIST NOISY DC={mnist_dc}_N={steps}_M={M}')
    return img, target
