import numpy as np 


def Forward(x, h_fft, toto):

    if toto == 1:
        out = np.real(np.fft.ifft2(np.fft.fft2(x)*h_fft))
    elif toto == 2:
        hc_fft = np.conj(h_fft)
        out = np.real(np.fft.ifft2(np.fft.fft2(x)*h_fft))
    return out


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