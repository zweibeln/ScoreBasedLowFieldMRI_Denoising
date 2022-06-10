import os
import sys
DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.insert(0, DIR)
from configs.set_up_variables import *
from models.init_model import *
from utils.draw_data import *

# Initialize model and relateds
model, optimizer, optimizer2, scheduler, last_ep = init_model()
model = model.to(device)
Loss = nn.MSELoss()

#### Load Data ####
observation_train, _, target_train = draw_data_train(set_type = 'train')
# observation_val, _, target_val = draw_data_train(set_type = 'val')

slice_loss = []
volume_loss_train = []
# volume_loss_val = []
iteration = 0
for observation_t,target_t in zip(observation_train,target_train):
    with tqdm.tqdm(total=len(observation_t), desc=f'Training') as pbar:
        for obs, tar in zip(observation_t, target_t):
            iteration += 1
            obs = obs.unsqueeze(0)
            tar = tar.unsqueeze(0)
            obs = obs - torch.min(obs)
            obs = obs/torch.max(obs)
            tar = tar - torch.min(tar)
            tar = tar/torch.max(tar)
            optimizer.zero_grad()
            output = model(obs.to(device))
            loss = torch.sqrt(Loss(output, tar.to(device)))
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
                plt.title('Supervised UNet Training Loss')
                plt.savefig(f'SupervisedLoss{iteration}.png')

        volume_loss_train.append(sum(slice_loss) / len(slice_loss))
        slice_loss = []
        print(f'TRAIN VOLUME LOSS = {volume_loss_train[-1]}')
    # with torch.no_grad(), tqdm.tqdm(total=len(observation_v), desc=f'Validation') as pbarv:
    #     for obs, tar in zip(observation_v, target_v):
    #         obs = obs.unsqueeze(0)
    #         tar = tar.unsqueeze(0)
    #         obs = obs - torch.min(obs)
    #         obs = obs/torch.max(obs)
    #         tar = tar - torch.min(tar)
    #         tar = tar/torch.max(tar)
    #         output = model(obs.to(device))
    #         loss = torch.sqrt(Loss(output.to(device), tar.to(device)))
    #         slice_loss.append(loss.item())
    #         pbarv.update(obs.shape[0])
        
    #     volume_loss_val.append(sum(slice_loss) / len(slice_loss))
    #     slice_loss = []
    #     print(f'VAL VOLUME LOSS = {volume_loss_val[-1]}')


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
plt.title('Supervised UNet Training Loss')
plt.savefig(f'SupervisedLoss{iteration}.png')





