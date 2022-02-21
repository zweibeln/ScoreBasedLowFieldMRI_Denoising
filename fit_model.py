from init_model import *
from set_up_variables import *
from ema import *
from score_loss import *


def fit_model(x_train):

    score_model, opt_score, opt_score2, scheduler, last_ep = init_model()
    print(f'{model_name} Model has been chosen')

    total_loss_tr = []
    epoch_loss_tr = []
    print(f'Total num of epochs = {num_epochs}')

    ema = EMA(ema_value)
    for name, param in score_model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    print(f'CURRENT USED DEVICE IS {device}!')

    # Training
    for epoch in range(num_epochs):
        print(f'Training Epoch {epoch + 1}/{num_epochs}')
        amount_tr = 0
        with tqdm.tqdm(total=len(x_train), desc=f'Training') as pbar:
            for x in x_train:
                # TRAINING AND STORING LOSS
                if last_ep < 4000:
                    opt_score.zero_grad()
                    loss = score_loss(score_model, x, device, sigma_max=sigma_max)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(score_model.parameters(), grad_clip_norm)
                    opt_score.step()
                else:
                    opt_score2.zero_grad()
                    loss = score_loss(score_model, x, device, sigma_max=sigma_max)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(score_model.parameters(), grad_clip_norm)
                    opt_score2.step()

                for name, param in score_model.named_parameters():
                    if param.requires_grad:
                        param.data = ema(name, param.data)

                # Store loss
                epoch_loss_tr.append(loss)
                # Store the metrics of this step
                step_metrics_tr = {'lossG': loss.item()}
                pbar.set_postfix(**step_metrics_tr)
                pbar.update(list(x.shape)[0])
                amount_tr = amount_tr + 1
        # lrs.append(opt_score.param_groups[0]["lr"])
        scheduler.step()

        tr_loss = sum(epoch_loss_tr[int(epoch * len(x_train)):int((epoch + 1) * len(x_train))]) / amount_tr
        print(f'Epoch Training Loss ={tr_loss}')
        total_loss_tr.append(tr_loss.to('cpu').detach().numpy())

    # Saving Model
    print('MODEL IS SAVED')
    torch.save({'model_state_dict': score_model.state_dict(),
                'adam_optimizer_state_dict': opt_score.state_dict(),
                'sgd_optimizer_state_dict': opt_score2.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, f'checkpoint_{model_name}_{set_type}.pth')

    return total_loss_tr
