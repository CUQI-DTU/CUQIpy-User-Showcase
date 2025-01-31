import torch
import cuqi
import numpy as np
from pathlib import Path
import pickle
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator

# smoothed Laplace prior 
class LaplaceSmoothed(cuqi.distribution.Distribution):
    def __init__(self, location, scale, beta, **kwargs):
        super().__init__(**kwargs)
        self.location = location
        self.scale = scale
        self.beta = beta
  
    def logpdf(self, x):
        if isinstance(x, (float, int, np.ndarray)):
            x = torch.tensor([x], dtype=torch.double)
        return torch.sum(torch.log(0.5/self.scale) - torch.sqrt((x-self.location)**2+self.beta)/self.scale)
    
    def gradient(self, x):
        x.requires_grad = True
        x.grad = None
        Q = self.logpdf(x)     # Forward pass
        Q.backward()           # Backward pass
        return x.grad

    def _sample(self,N=1,rng=None):
        return None

# load files
def load(file):
    file = Path(file)
    file = open(file, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable

# pickle files
def save(file, variable):
    file = Path(file)
    file = open(file, 'wb')
    pickle.dump(variable, file)
    file.close()

# fast exponential and Laplace sampler, samples product-form densities with different rate parameters
# see, e.g., https://www.johndcook.com/blog/2018/03/13/generating-laplace-random-variables/
fast_exponential = lambda rate, N: - np.log( np.random.random( size=(rate.size, N) ) ) / rate[:, None]
fast_Laplace = lambda rate, N: fast_exponential(rate, N) - fast_exponential(rate, N)

# samples from a normal density with inverse covariance = lam* A^T A + D and mean = mu
# here A is a linear operator (the forward model) and D is a diagonal matrix (here given by the vector on the main diagonal)
def linear_RTO(A, A_adj, lam, D, mu, N):
    m = A(mu).size
    d = mu.size
    sqrt_lam = np.sqrt(lam)
    sqrt_D = np.sqrt(D)
    matvec = lambda x: np.concatenate((sqrt_lam*A(x), sqrt_D*x))
    rmatvec = lambda y: sqrt_lam*A_adj(y[:m]) + sqrt_D*y[m:]
    C_adj = LinearOperator(shape=(m+d, d), matvec=matvec, rmatvec=rmatvec)
    C_adj_mu = C_adj(mu)
    samples = np.zeros((d, N))
    for ii in range(N):
        b = C_adj_mu + np.random.normal(size=d+m)
        x, istop, itnr, normr = lsqr(C_adj, b, x0=mu, atol=1e-6, btol=1e-6)[:4]
        print('\r', f'it {ii+1}, istop {istop}', end='')
        samples[:, ii] = x
    return samples