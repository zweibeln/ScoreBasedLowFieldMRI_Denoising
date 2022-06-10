from configs.imports import *


def snr_calculate(img):
    dimy = img.shape[-1]
    dimx = img.shape[-2]
    mid_pt = [int(np.floor(dimx/2)), int(np.floor(dimy/2))]
    xlb = int(mid_pt[0] - np.floor(mid_pt[0] / 2))
    xub = int(mid_pt[0] + np.floor(mid_pt[0] / 2))
    ylb = int(mid_pt[1] - np.floor(mid_pt[1] / 2))
    yub = int(mid_pt[1] + np.floor(mid_pt[1] / 2))
    signal_pow = torch.mean(img[xlb:xub,ylb:yub])

    noise_box_side_length = int(np.floor(np.max((dimy,dimx))/16))
    #corner1 = img[0:noise_box_side_length, 0:noise_box_side_length]
    corner2 = img[0:noise_box_side_length, dimy-noise_box_side_length:dimy]
    corner3 = img[dimx-noise_box_side_length:dimx, 0:noise_box_side_length]
    # corner4 = img[dimx-noise_box_side_length:dimx, dimy-noise_box_side_length:dimy]
    noise_pow = (torch.std(corner2)+torch.std(corner3))/2 #+torch.std(corner1)+torch.std(corner4))/4

    return (signal_pow/noise_pow)*0.66, noise_pow

