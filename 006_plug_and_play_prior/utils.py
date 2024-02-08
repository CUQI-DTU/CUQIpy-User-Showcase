import torch 
import torch.nn as nn
import numpy as np

def center_kernel(kernel, s):
    """
    This function allows us to center the kernel at the top-left position.
    It means that the 0 frequency is located at the top-left postion of the image.
    Inputs:
        -kernel: the convolution kernel with the shape as the image we are dealing with.
        -l (int): kernel_size = 2*l+1
    Output:
        -centered_kernel
    """
    centered_kernel = np.zeros(kernel.shape)

    centered_kernel[0:s+1,0:s+1] = kernel[s:2*s+1, s:2*s+1]
    centered_kernel[0:s+1,-s:] = kernel[s:2*s+1, 0:s]
    centered_kernel[-s:,0:s+1] = kernel[0:s, s:2*s+1]
    centered_kernel[-s:,-s:] = kernel[0:s, 0:s]

    return centered_kernel

class DnCNN(nn.Module):
    def __init__(self, nc_in, nc_out, depth, act_mode, bias=True, nf=64):
        super(DnCNN, self).__init__()

        self.depth = depth

        self.in_conv = nn.Conv2d(nc_in, nf, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(self.depth - 2)])
        self.out_conv = nn.Conv2d(nf, nc_out, kernel_size=3, stride=1, padding=1, bias=bias)

        if act_mode == 'R':  # Kai Zhang's nomenclature
            self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

    def forward(self, x_in):

        x = self.in_conv(x_in)
        x = self.nl_list[0](x)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x)
            x = self.nl_list[i + 1](x_l)

        return self.out_conv(x) + x_in

def pytorch_denoiser(xtilde, model, device = 'cpu'):
    """
    pytorch_denoiser: to convert a numpy array to torch ten back to numpy
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


def load_dncnn_weights(n_ch, n_lev, ljr, device, path = 'ckpts/finetuned/'):

    name_file = 'DNCNN_nch_' + str(n_ch) + '_sigma_' + str(n_lev) + '_ljr_' + str(ljr)
    
    model_weights = torch.load(path + name_file + '.ckpt', map_location = torch.device(device if torch.cuda.is_available() else "cpu"))

    cuda = True if torch.cuda.is_available() else False
    #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    avg, bn, depth = False, False, 20
    net = DnCNN(1, 1, depth, 'R')

    if cuda:
        model = nn.DataParallel(net, device_ids = [int(str(device)[-1])], output_device = device)
    else:
        model = nn.DataParallel(net)

    model.module.load_state_dict(model_weights["state_dict"], strict=True)
    model.eval() 
    model.to(device)
    
    return model