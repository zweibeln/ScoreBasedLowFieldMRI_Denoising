# import os
# DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# sys.path.insert(0, DIR)
import sys
from utils.draw_data import *
from training.fit_model import *

# Load data to train score model    
_, _, x_train = draw_data_train(set_type='train', low_f=1.49, high_f=1.49)
# x_train = [x.to(device) for x in x_train]

total_loss_tr = fit_model(x_train)

plt.figure()
plt.plot(total_loss_tr, 'b-', label='ScoreFunctionLoss_Train')
plt.legend()
plt.title('Score Loss')
plt.xlabel('x1000 iterations')
plt.savefig(f'ScoreLoss_{model_name}_{set_type}.png')
