from configs.set_up_variables import *
from mpl_toolkits.axes_grid1 import ImageGrid
from losses.ssim_loss import *
from utils.snr_calculate import *


def plot_images(x, statement='NOISY'):
    num_of_images = x.shape[0]
    fig = plt.figure(figsize=(20., 20.))
    plt.style.use('grayscale')
    plt.title(f'{set_type}_{statement}_{dc_f[0]}T')
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(int(np.sqrt(num_of_images)), int(np.sqrt(num_of_images))),
                     axes_pad=0.,  # pad between axes in inch.
                     )
    for idx, ax in enumerate(grid):
        ax.imshow(x[idx, 0, :, :].cpu().detach().numpy())
        ax.set_xticks([])
        ax.set_yticks([])

    if statement != 'GND' or 'NOISY':
        plt.savefig(f'{set_type}_{statement}_{model_name}_dc_{data_consistency}_N={steps}_M={M}.png')
    else:
        plt.savefig(f'{set_type}_{statement}_{dc_f[0]}T.png')
    print(f'SAVED AS {set_type}_{statement}_{model_name}_dc_{data_consistency}_N={steps}_M={M}')


