from imports import *


def dc_in_mri(img, mask, device='cuda' if torch.cuda.is_available() else 'cpu'):
    img = img.to('cpu')
    mask = mask.to('cpu')
    k_space = FT.fftn(img, s=None, dim=(-2, -1), norm=None)
    k_space_masked = (mask * k_space).to(device)
    img_masked = torch.abs(FT.ifftn(k_space_masked, s=None, dim=(-2, -1), norm=None)).to(device)
    img = img.to(device)
    return img, img_masked
