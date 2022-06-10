import os
import sys
DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.insert(0, DIR)
from configs.set_up_variables import *
from models.init_model import *
from utils.draw_data import *

model, optimizer, optimizer2, scheduler, last_ep = init_model()
Loss = nn.MSELoss()

observation, mask, target = draw_data_train(set_type = 'train')

slice_loss = []
volume_loss_train = []
# volume_loss_val = []
iteration = 0
for observation_t,mask_t,target_t in zip(observation,mask,target):
    with tqdm.tqdm(total=len(observation_t), desc=f'Training') as pbar:
        for obs, msk, tar in zip(observation_t, mask_t, target_t):
            x = torch.zeros((1,1,128,128))
            iteration += 1
            obs = obs.unsqueeze(0)
            tar = tar.unsqueeze(0)
            obs = obs - torch.min(obs)
            obs = obs/torch.max(obs)
            tar = tar - torch.min(tar)
            tar = tar/torch.max(tar)
            optimizer.zero_grad()
            obs_f = reorganize(FT.fftn(obs, s=None, dim=(-2, -1), norm=None))
            output = model(x, msk, obs_f)
            loss = torch.sqrt(Loss(output.to(device), tar.to(device)))
            loss.backward()
            optimizer.step()
            slice_loss.append(loss.item())
            pbar.update(obs.shape[0])
            scheduler.step()
            if iteration%250 == 0:
                print(f'CURRENT ITERATION = {iteration}')
                print('MODEL IS SAVED')
                torch.save({'model_state_dict': model.state_dict(),
                'adam_optimizer_state_dict': optimizer.state_dict(),
                'sgd_optimizer_state_dict': optimizer2.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, f'checkpoint_{model_name}_{set_type}.pth')

            if iteration%5000 == 0:
                print(f'CURRENT ITERATION = {iteration}')
                plt.figure()
                plt.plot(volume_loss_train, label='Train Loss')
                # plt.plot(volume_loss_val, label='Val Loss')
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('RMSE')
                plt.title('Unfolded Training Loss')
                plt.savefig(f'UnfoldedLoss{iteration}.png')

        volume_loss_train.append(sum(slice_loss) / len(slice_loss))
        slice_loss = []
        print(f'TRAIN VOLUME LOSS = {volume_loss_train[-1]}')

print('MODEL IS SAVED')
torch.save({'model_state_dict': model.state_dict(),
            'adam_optimizer_state_dict': optimizer.state_dict(),
            'sgd_optimizer_state_dict': optimizer2.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()}, f'checkpoint_{model_name}_{set_type}.pth')

plt.figure()
plt.plot(volume_loss_train, label='Train Loss')
# plt.plot(volume_loss_val, label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Unfolded Training Loss')
plt.savefig(f'UnfoldedLoss{iteration}.png')
