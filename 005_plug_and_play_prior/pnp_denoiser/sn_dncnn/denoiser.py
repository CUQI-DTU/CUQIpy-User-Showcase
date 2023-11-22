"""
    simplified interface to the pretrained denoisers in Ryu (2019)
    author: A. Almansa
    date: 2020-01-08
"""

import numpy as np
import torch


def pytorch_denoiser_residual(xtilde, model, device = 'cpu'):
    """
    pytorch_denoiser
    Inputs:
        xtilde      noisy image
        model       pytorch denoising model
        device
    Output:
        x           denoised image
    """


    # image size
    N = np.shape(xtilde)
    n, m = int(np.sqrt(N)), int(np.sqrt(N))    

    with torch.no_grad():
        # load to torch
        xtilde_torch = np.reshape(xtilde, (1,1,m,n))
        if device == "cpu":
            xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor)
        else:
            xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.cuda.FloatTensor).to(device)
            #xtilde_torch.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        
        # denoise
        #This line allows to compute a Gpu Float-tensor into a cpu float tensor and
        # then into a numpy array
        r = model(xtilde_torch)
        #r = np.reshape(r, -1)
        x = xtilde_torch - r
        x = x.cpu().numpy()


    # Reshape back
    x = np.reshape(x,(m,n))
    x = x.flatten()

    return x

def pytorch_denoiser(xtilde, model, device = 'cpu'):
    """
    pytorch_denoiser
    Inputs:
        xtilde      noisy vector to convert into an image
        model       pytorch denoising model
        device      "cpu" or "cuda:0"

    Output:
        x           denoised image
    """

    # image size


    N = np.shape(xtilde)
    n, m = int(np.sqrt(N)), int(np.sqrt(N))    

    with torch.no_grad():
        # load to torch
        xtilde_torch = np.reshape(xtilde, (1,1,m,n))
        if device == "cpu":
            xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor)
            #xtilde_torch.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        else:
            xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.cuda.FloatTensor).to(device)

        # denoise
        #This line allows to compute a Gpu Float-tensor into a cpu float tensor and
        # then into a numpy array
        x = model(xtilde_torch).cpu().numpy()
        #r = np.reshape(r, -1)

    # Reshape back
    x = np.reshape(x,(m,n))
    x = x.flatten()

    return x


def denoiser_residual(x,model):
    """
    pytorch_denoiser
    Inputs:
        xtilde      noisy tensor
        model       pytorch denoising model

    Output:
        x           denoised tensor

    """

    # denoise
    with torch.no_grad():
        #xtorch = xtilde.unsqueeze(0).unsqueeze(0)
        r = model(x)
        #r = np.reshape(r, -1)
        out = x - r

        #out = torch.squeeze(out)

    return out