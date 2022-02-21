from set_up_variables import *


def print_configs():
    print(f'Selected model is {model_name}')

    print(f'Set Type = {set_type}')
    print(f'Data Consistency = {data_consistency}')
    print(f'Selected Device = {device}')

    print(f't0 = {t_0}')
    print(f'T = {Tmax}')
    print(f'Number of Predictor Steps (N) = {steps}')
    print(f'Number of Corrector Steps (M) = {M}')
    print(f'SNR = {r}')

    print(f'sigma_min = {sigma[0]}')
    print(f'sigma_max = {sigma[-1]}')

    print(f'Sampling Langevin = {langevin}')
