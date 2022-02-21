from imports import *


class SSIM_loss(torch.nn.Module):  # Pixel values upto 255

    def __init__(self):
        super().__init__()

    def forward(self, img_rec, img_tar):
        img_rec = img_rec[0,0,:,:]-torch.min(img_rec[0,0,:,:])
        img_rec = 255*(img_rec/torch.max(img_rec))
        img_tar = 255*(img_tar[0,0,:,:]/torch.max(img_tar))
        sizes = img_tar.shape
        win_sz = torch.tensor(16)
        iter_range = int(torch.div(sizes[-1],win_sz))
        k1 = torch.tensor(0.01)
        k2 = torch.tensor(0.03)
        L = torch.add(torch.pow(torch.tensor(2),torch.tensor(8)),-1)
        c1 = torch.pow(torch.multiply(k1,L),2)
        c2 = torch.pow(torch.multiply(k2,L),2)
        ssim_cur = 0
        ssim_acc = 0
        for ii in range(iter_range):
            for j in range(iter_range):
                # get the first window
                rec_w = img_rec[ii*win_sz:(ii+1)*win_sz,j*win_sz:(j+1)*win_sz]
                tar_w = img_tar[ii*win_sz:(ii+1)*win_sz,j*win_sz:(j+1)*win_sz]

                # calculate the metrics
                mu_rec   = torch.mean(rec_w)
                mu_tar   = torch.mean(tar_w)
                var_rec  = torch.var(rec_w, unbiased=False)
                var_tar  = torch.var(tar_w, unbiased=False)
                cros_var = torch.mean((rec_w-mu_rec)*(tar_w-mu_tar)) # torch.transpose(,0,1)

                nom1   = 2*mu_rec*mu_tar + c1  # torch.add(torch.multiply(torch.multiply(torch.tensor(2),mu_rec),mu_tar),c1)
                nom2   = 2*cros_var + c2  #torch.add(torch.multiply(torch.tensor(2),cros_var),c2)
                denom1 = torch.pow(mu_rec, 2) + torch.pow(mu_tar, 2) + c1  #torch.add(torch.add(torch.multiply(mu_rec,mu_rec),torch.multiply(mu_tar,mu_tar)),c1)
                denom2 = var_rec + var_tar + c2  #torch.add(torch.add(var_rec,var_tar),c2)

                # SSIM metric
                nom = nom1*nom2#torch.multiply(nom1,nom2)
                denom = denom1*denom2#torch.multiply(denom1,denom2)
                ssim_cur = nom/denom#torch.div(nom,denom)  # current
                ssim_acc = ssim_cur+ssim_acc#torch.add(ssim_acc,ssim_cur)  # accumulation

        #DSIM = torch.add(1,torch.multiply(torch.div(ssim_acc, iter_range),-1))
        ssim_acc = ssim_acc/pow((sizes[-1]/win_sz),2)
        DSIM = (1.-ssim_acc)/2.
        return DSIM