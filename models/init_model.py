from configs import imports, set_up_variables
# from ncsnpp_try import *
# from score_unet import *
# from score_unet_enc import *
# from ncsnpp_try_aapool import *
# from tutorial_code import *
# from ncsnpp_try_mpool import *
# from ncsnpp_try_gnorm import *
# from DnCNN import *
# from ncsnpp_wot import *
from models.unet import *
from models.unet_wot import *
# from models.unet_wot_cd import *
from models.unfolded import *
# from ncsnv2 import *
# from unet_wot_stack import *
# from unet_wot_mp import *
# from ResDnCNN import *


def init_model():
    name = model_name
    print('Model Configurations: ')
    print(f'Num of Classes = {num_class}')
    print(f'Output Size = {out_sz}')
    # if name == 'ncsnpp':
    #     print(f'{name} model is initialized')
    #     enc_chs=(16, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64, 16)
    #     score_model = ScoreBigGAN(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights, set_type=set_type)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    # if name == 'ncsnv2':
    #     print(f'{name} model is initialized')
    #
    #     score_model =  NCSNv2_256(set_type=set_type)
    #     score_model = score_model.to(device)
    # if name == 'ncsnpp_gn':
    #     print(f'{name} model is initialized')
    #     enc_chs=(16, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64, 16)
    #     score_model = ScoreBigGAN_GN(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights, set_type=set_type)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    # if name == 'ncsnpp_aapool':
    #     print(f'{name} model is initialized')
    #     enc_chs=(16, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64, 16)
    #     score_model = ScoreBigGANAAPool(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    # if name == 'ncsnpp_mpool':
    #     print(f'{name} model is initialized')
    #     enc_chs=(16, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64, 16)
    #     score_model = ScoreBigGANMPool(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    # if name == 'ncsnpp_wot':
    #     print(f'{name} model is initialized')
    #     enc_chs=(16, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64, 16)
    #     score_model = NCSNPP_WOT(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    # if name == 'unet_wot_mp':
    #     print(f'{name} model is initialized')
    #     enc_chs=(1, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64)
    #     score_model = UNET_WOT_MP(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights, set_type=set_type)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    # if name == 'tutorial_code':
    #     print(f'{name} model is initialized')
    #     score_model = ScoreNet()
    #     score_model = score_model.to(device)
    # if name == 'DnCNN':
    #     print(f'{name} model is initialized')
    #     score_model = DnCNN(set_type=set_type)
    #     score_model = score_model.to(device)
    # if name == 'ResDnCNN':
    #     print(f'{name} model is initialized')
    #     score_model = ResDnCNN(set_type=set_type)
    #     score_model = score_model.to(device)
    # if name == 'unet':
    #     print(f'{name} model is initialized')
    #     enc_chs=(1, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64)
    #     score_model = ScoreUNet(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    # if name == 'unet_enc':
    #     print(f'{name} model is initialized')
    #     enc_chs=(1, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64)
    #     score_model = ScoreUNet_Enc(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    if name == 'unet_wot':
        print(f'{name} model is initialized')
        score_model = UNET_WOT(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
                                out_sz=out_sz, initialize_weights=initialize_weights, set_type='mri')
        score_model = score_model.to(device)
        print(f'Depth Of Model = {enc_chs[-1]}')

    # if name == 'unet_wot_cd':
    #     print(f'{name} model is initialized')
    #     score_model = UNET_WOT_CD(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights, set_type='mri')
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')

    if name == 'unet':
        print(f'{name} model is initialized')
        score_model = UNET(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
                                out_sz=out_sz, initialize_weights=initialize_weights, set_type='mri')
        score_model = score_model.to(device)
        print(f'Depth Of Model = {enc_chs[-1]}')

    if name == 'unfolded':
        print(f'{name} model is initialized')
        score_model = Unfolded()
        score_model = score_model.to(device)


    # if name == 'unet_wot_stack':
    #     print(f'{name} model is initialized')
    #     enc_chs=(1, 64, 128, 256, 512, 1024)  # Encoder output channels, the first element is the channel size of the input
    #     dec_chs=(1024, 512, 256, 128, 64)
    #     repetition = 2
    #     score_model = UNET_WOT_STACK(enc_chs=enc_chs, dec_chs=dec_chs, num_class=1, retain_dim=retain_dim,
    #                             out_sz=out_sz, initialize_weights=initialize_weights,repetition=repetition)
    #     score_model = score_model.to(device)
    #     print(f'Depth Of Model = {enc_chs[-1]}')
    #     print(f'Repetition = {repetition}')

    opt_score = optim.Adam(score_model.parameters(), lr=adam_lr, betas=betas)
    opt_score2 = optim.SGD(score_model.parameters(), lr=sgd_lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_score, milestones=milestones, gamma=gamma)

    if os.path.exists(os.getcwd() + '/' + f'checkpoint_{model_name}_mri.pth'):
        print(f'checkpoint_{model_name}.pth IS FOUND AND STATE DICTS ARE LOADED')
        checkpoint = torch.load(f'checkpoint_{model_name}_mri.pth', map_location=device)
        score_model.load_state_dict(checkpoint['model_state_dict'])
        opt_score.load_state_dict(checkpoint['adam_optimizer_state_dict'])
        opt_score2.load_state_dict(checkpoint['sgd_optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_ep = checkpoint['scheduler_state_dict']['last_epoch']
        print(f'LAST EPOCH = {last_ep}')
        score_model.eval()
    else:
        print(f'NO SAVED {model_name}_mri FOUND SO NEW ONE INITIALIZED')
        last_ep = 0

    return score_model, opt_score, opt_score2, scheduler, last_ep