from set_up_variables import *
from produce_data import *


def arrange_data_for_mri_sampling():
    # Load data to train score model
    x_train, h_train, y_t = produce_data_sampling(field_str_tr=dc_f)
    print(f'MRI DATA FIELD STRENGTH {dc_f[0]}T')
    print('MRI DATA FOR SAMPLING HAS BEEN PRODUCED')
    x_train = [x.to(device) for x in x_train]
    y_t = [y.to(device) for y in y_t]
    h_train = [h.to(device) for h in h_train]

    img = x_train[img_nums]  # Image to be reconstructed
    target = y_t[img_nums]  # Target
    mask = h_train[img_nums]  # Mask of corresponding image
    return img, mask, target
