
import torch 
import torch.nn as nn


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


def load_dncnn_weights(n_ch, n_lev, ljr, device):

    name_file = 'DNCNN_nch_' + str(n_ch) + '_sigma_' + str(n_lev) + '_ljr_' + str(ljr)
    
    model_weights = torch.load('ckpts/finetuned/' + name_file + '.ckpt', map_location = torch.device(device if torch.cuda.is_available() else "cpu"))

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

def load_dncnn_weights_2(n_ch, n_lev, ljr, device, path = 'ckpts/finetuned/'):

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