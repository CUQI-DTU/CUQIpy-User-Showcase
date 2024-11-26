import torch
import cuqi
import numpy as np
from pathlib import Path
import pickle
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

# load and pickle files
def load(file):
    file = Path(file)
    file = open(file, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable

def save(file, variable):
    file = Path(file)
    file = open(file, 'wb')
    pickle.dump(variable, file)
    file.close()

# fast exponential and Laplace sampler, samples product-form densities with different rate parameters
# see, e.g., https://www.johndcook.com/blog/2018/03/13/generating-laplace-random-variables/
fast_exponential = lambda rate, N: - np.log( np.random.random( size=(rate.size, N) ) ) / rate[:, None]
fast_Laplace = lambda rate, N: fast_exponential(rate, N) - fast_exponential(rate, N)