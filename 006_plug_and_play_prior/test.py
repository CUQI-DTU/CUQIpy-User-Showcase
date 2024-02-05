# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

import cuqi
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian, Posterior
from cuqi.problem import BayesianProblem
from cuqi.sampler import LinearRTO

from ConstrainedLinearRTO import ImplicitRegularizedGaussian, ProjectNonnegative, ProjectBox, FISTA, RegularizedLinearRTO
from utils import *

# Define a Linear operator 



#%% Load problem
N = 64 # image size
l = 2 # box-filter kernel of size 2*l+1
lambd = 1000
sigma = np.sqrt(1/lambd) # standard deviation of the noise
gamma = sigma/1000
sigma2 = sigma**2
gamma2 = gamma**2
# Loading the image
I = plt.imread("simpson_nb512.png")
im = I[2*N:3*N, N:2*N]
im /= np.max(im)
# Seed
np.random.seed(0)

# Observation model
l_k = 2*l + 1
h_ = np.ones((l_k, l_k))/(l_k)**2
k_uniform = np.zeros(im.shape)
k_uniform[0:l_k,0:l_k] = np.ones((l_k, l_k))/(l_k)**2
k_uniform = center_kernel(k_uniform, l)
h_fft = np.fft.fft2(k_uniform)
hc_fft = np.conj(h_fft)


A = lambda x : np.real(np.fft.ifft2(np.fft.fft2(x)*h_fft))
AT = lambda x : np.real(np.fft.ifft2(np.fft.fft2(x)*hc_fft))
geometry = cuqi.geometry.Image2D((N,N))
A_op = LinearModel(A, AT, range_geometry = geometry, domain_geometry = geometry)

# Observation
X_generation = Gaussian(0, gamma2, geometry = geometry)
Y_generation = Gaussian(A_op @ X_generation, sigma2, geometry = geometry)
y_data = Y_generation(X_generation = im).sample()
y_data.plot()
plt.show()

#%% Load prior denoiser
prior_denoiser = "fine" # "sn_dncnn" or "fine"
device = "cuda:0"
if prior_denoiser == "sn_dncnn":
    sys.path.append("./pnp_denoiser/sn_dncnn/")
    from ryu_utils.utils import load_model
    from denoiser import pytorch_denoiser_residual
    if device == "cpu":
        cuda = False
    else:
        cuda = True
    s = 40
    delta = 1/(1/sigma2 + 1/gamma2)
    model = load_model(model_type = "RealSN_DnCNN", sigma = s, device = device, cuda = cuda, path = "pnp_denoiser/sn_dncnn/Pretrained_models/")
    Ds = lambda x, z : pytorch_denoiser_residual(x, model, device)

elif prior_denoiser == "fine":
    sys.path.append("./pnp_denoiser/sn_dncnn/")
    sys.path.append("./pnp_denoiser/fine/")
    from utils_terris_dncnn import load_dncnn_weights_2
    from denoiser import pytorch_denoiser
    if device == "cpu":
        cuda = False
    else:
        cuda = True
    s = 2.5
    n_ch = 1
    if s == 2.5:
        ljr = 0.005
    elif s == 2.25:
        ljr = 0.002
    delta = 0.5/(1/sigma2 + 1/gamma2)
    model = load_dncnn_weights_2(n_ch, s, ljr, device, path = 'pnp_denoiser/fine/ckpts/finetuned/')
    Ds = lambda x, z : pytorch_denoiser(x, model, device)

#%% Bayesian problem formulation
X  = ImplicitRegularizedGaussian(Gaussian(0, gamma2, geometry = geometry), proximal = Ds)
Y = Gaussian(A_op @ X, sigma2, geometry = geometry)
BP = BayesianProblem(Y, X).set_data(Y = y_data)
posterior = BP.posterior

sampler = RegularizedLinearRTO(posterior, maxit = 200, stepsize = delta, abstol=1e-10)
samples = sampler.sample(5, 0)


# plt.figure()
# samples.plot_median()
# samples.plot_ci()
# plt.plot(x_true)
# plt.legend(["mean", "CI", "truth"])
# plt.show()



#%%
"""
#%% Compute constrained maximum likelihood
projector = lambda z, gamma: ProjectNonnegative(z)
solver = FISTA(Amat, y_data + 0.01*random.normal(size = y_data.shape), np.zeros(n), projector,
               maxit = 100, stepsize = 5e-1, abstol=1e-10, adaptive = False)
sol, k = solver.solve()

plt.plot(x_space, x_true)
plt.plot(x_space, sol)
plt.title(f"Nonnegative maximum likelihood, iterations = {k}")

#%% Perform UQ

A  = LinearModel(Amat)
x  = Gaussian(0.5*np.ones(n), 0.01)
y  = Gaussian(A@x, 0.001)
BP = BayesianProblem(y, x).set_data(y=y_data)
posterior = BP.posterior

sampler = LinearRTO(posterior)
samples = sampler.sample(1000, 100)

plt.figure()
samples.plot_median()
samples.plot_ci()
plt.plot(x_true)
plt.legend(["mean", "CI", "truth"])
plt.show()

projector = lambda z: ProjectNonnegative(z)
sampler = ConstrainedLinearRTO(posterior, projector, maxit=100, stepsize = 5e-4, abstol=1e-10)
samples = sampler.sample(1000, 100)

plt.figure()
samples.plot_median()
samples.plot_ci()
plt.plot(x_true)
plt.legend(["median", "CI", "mean", "truth"])
plt.show()

projector = lambda z: ProjectBox(z)
sampler = ConstrainedLinearRTO(posterior, projector, maxit=100, stepsize = 5e-4, abstol=1e-10)
samples = sampler.sample(1000, 100)

plt.figure()
samples.plot_median()
samples.plot_ci()
plt.plot(x_true)
plt.legend(["median", "CI", "mean", "truth"])
plt.show()

#%%

proximal = lambda z, gamma: ProjectNonnegative(z)
sampler = RegularizedLinearRTO(posterior, proximal, maxit=100, stepsize = 5e-4, abstol=1e-10)
samples = sampler.sample(1000, 100)

plt.figure()
samples.plot_median()
samples.plot_ci()
plt.plot(x_true)
plt.legend(["median", "CI", "mean", "truth"])
plt.show()





posterior = BP.posterior

sampler = LinearRTO(posterior)
samples = sampler.sample(1000, 100)

plt.figure()
samples.plot_median()
samples.plot_ci()
plt.plot(x_true)
plt.legend(["mean", "CI", "truth"])
plt.show()

sampler = ConstrainedLinearRTO(posterior, projector, maxit=100, stepsize = 5e-4, abstol=1e-10)
samples = sampler.sample(1000, 100)
"""