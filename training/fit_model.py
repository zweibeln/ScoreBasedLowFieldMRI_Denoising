import os
import sys
DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.insert(0, DIR)
from models.init_model import *
from configs.set_up_variables import *
from training.ema import *
from losses.score_loss import *


def fit_model(x_train):

    score_model, opt_score, opt_score2, scheduler, last_ep = init_model()
    print(f'{model_name} Model has been chosen')

    total_loss_tr = []

    ema = EMA(ema_value)
    for name, param in score_model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    print(f'CURRENT USED DEVICE IS {device}!')

    # Training
    volume_loss = []
    slice_loss = []
    iteration_loss = []
    
    iteration = 1 
    for _ in range(2):
        for y in x_train:  # 1 list out of 973
            with tqdm.tqdm(total=len(y), desc=f'Training') as pbar:
                for y_tr in y:
                    y_tr = y_tr - torch.min(y_tr)
                    y_tr = y_tr / torch.max(y_tr)
                    y_tr = y_tr.unsqueeze(0)
                    iteration += 1 
                    if iteration%250 == 0:
                        # Saving Model
                        print(f'Model Is Saved at ITERATION {iteration}')
                        torch.save({'model_state_dict': score_model.state_dict(),                                                 
                        'adam_optimizer_state_dict': opt_score.state_dict(),
                        'sgd_optimizer_state_dict': opt_score2.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, f'checkpoint_{model_name}_{set_type}.pth')

                    # TRAINING AND STORING LOSS
                    opt_score.zero_grad()
                    loss = score_loss(score_model, y_tr, device, sigma_max=sigma_max)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(score_model.parameters(),1)
                    opt_score.step()

                    for name, param in score_model.named_parameters():
                        if param.requires_grad:
                            param.data = ema(name, param.data)

                    if iteration%5000 == 0:
                        torch.set_grad_enabled(False)
                        x = torch.randn(9,1,128,128)
                        for idx in range(0,9):
                            x[idx,:,:,:] = x[idx,:,:,:]- torch.min(x[idx,:,:,:])
                            x[idx,:,:,:] = x[idx,:,:,:]/torch.max(x[idx,:,:,:])
                        with tqdm.tqdm(total=2000, desc=f'SDE') as pbar2:
                            for i in reversed(range(1,2000)):
                                score = score_model(x.to(device), dt[i].to(device)).to('cpu')  
                                # Predictor Step REAL
                                x = x.to('cpu')
                                x_mean = x + (sigma[i] ** 2 - sigma[i - 1] ** 2) * score
                                z = torch.randn(x.shape).to('cpu')
                                x = x_mean + torch.sqrt(sigma[i] ** 2 - sigma[i - 1] ** 2) * z
                                # Corrector Step REAL
                                z = torch.randn(x.shape).to('cpu')
                                z_norm = torch.sqrt(torch.sum(z[:, 0, :, :] ** 2))
                                s_norm = torch.sqrt(torch.sum(score[:, 0, :, :] ** 2))
                                epsilon = 2 * ((r * z_norm / s_norm) ** 2)
                                x_mean = x + epsilon * score
                                x = x_mean+ torch.sqrt(2 * epsilon) * z
                                pbar2.update(1)

                        x = x_mean.to('cpu')
                        num_of_images = x.shape[0]
                        fig = plt.figure(figsize=(20., 20.))
                        plt.style.use('grayscale')
                        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                        nrows_ncols=(int(np.sqrt(num_of_images)), int(np.sqrt(num_of_images))),
                                        axes_pad=0.,  # pad between axes in inch.
                                        )
                        for idx, ax in enumerate(grid):
                            ax.imshow(x[idx, 0, :, :].cpu().detach().numpy())
                            ax.set_xticks([])
                            ax.set_yticks([])

                        plt.savefig(f'UnCondSample_Iter{iteration}.png')
                        torch.set_grad_enabled(True)

                    # Store the metrics of this step
                    step_metrics_tr = {'loss': loss.item()}
                    pbar.set_postfix(**step_metrics_tr)
                    pbar.update(list(y_tr.shape)[0])
                    slice_loss.append(loss.item())
                    scheduler.step()
                    if iteration%5000 == 0:
                        print(f'LOSS PLOTS SAVED AT ITERATION {iteration}')
                        plt.figure()
                        plt.plot(volume_loss, 'b-', label='ScoreFunctionLoss_Train')
                        plt.legend()
                        plt.title('Score Loss')
                        plt.xlabel('Volume')
                        plt.savefig(f'ScoreLoss_Volume_{len(volume_loss)}.png')

                        plt.figure()
                        plt.plot(iteration_loss, 'b-', label='ScoreFunctionLoss_Train')
                        plt.legend()
                        plt.title('Score Loss')
                        plt.xlabel('x 100 Iteration')
                        plt.savefig(f'ScoreLoss_Iteration_{iteration}.png')
                volume_loss.append(sum(slice_loss)/len(slice_loss))
                slice_loss = []
                print(f'LOSS = {volume_loss[-1]}')
                if iteration%1000 == 0:
                    iteration_loss.append(sum(volume_loss)/len(volume_loss))

    plt.figure()
    plt.plot(volume_loss, 'b-', label='ScoreFunctionLoss_Train')
    plt.legend()
    plt.title('Score Loss')
    plt.xlabel('Volume')
    plt.savefig(f'ScoreLoss_Volume_{len(volume_loss)}_Rician.png')

    plt.figure()
    plt.plot(iteration_loss, 'b-', label='ScoreFunctionLoss_Train')
    plt.legend()
    plt.title('Score Loss')
    plt.xlabel('x 1000 Iteration')
    plt.savefig(f'ScoreLoss_Iteration_{iteration}_Rician.png')


    torch.save({'model_state_dict': score_model.state_dict(),                                                 
                'adam_optimizer_state_dict': opt_score.state_dict(),
                'sgd_optimizer_state_dict': opt_score2.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, f'checkpoint_{model_name}_{set_type}.pth')

    return iteration_loss
