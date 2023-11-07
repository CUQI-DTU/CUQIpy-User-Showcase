import scipy as sp
import numpy as np
from numpy import linalg as LA
import cuqi
from cuqi.sampler import Sampler
from cuqi.operator import Operator
from cuqi.distribution import Distribution, Gaussian
from cuqi.solver import CGLS

def ProjectNonnegative(x):
    return np.maximum(x, 0)

def ProjectBox(x, lower = None, upper = None):
    if lower is None:
        lower = np.zeros_like(x)
    
    if upper is None:
        upper = np.ones_like(x)
    
    return np.minimum(np.maximum(x, lower), upper)

def ProximalL1(x, gamma):
    """Proximal operator of ||x||_1, also known as the shrinkage of soft thresholding operator"""
    return np.multiply(np.sign(x), np.maximum(np.abs(x)-gamma, 0))

class FISTA(object):
    """Fast Iterative Shrinkage-Thresholding Algorithm for regularized least squares problems.

    Minimize ||Ax-b||^2 + f(x).
    
    Parameters
    ----------
    A : ndarray or callable f(x,*args).
    b : ndarray.
    x0 : ndarray. Initial guess.
    proximal : callable f(x, gamma) for proximal mapping.
    maxit : The maximum number of iterations.
    stepsize : The stepsize of the gradient step.
    abstol : The numerical tolerance for convergence checks.
    adapative : Whether to use FISTA or ISTA.
    """  
    def __init__(self, A, b, x0, proximal, maxit = 100, stepsize = 1e0, abstol = 1e-14, adaptive = False):
        
        self.A = A
        self.b = b
        self.x0 = x0
        self.proximal = proximal
        self.maxit = int(maxit)
        self.stepsize = stepsize
        self.abstol = abstol
        self.adaptive = adaptive
        if not callable(A):
            self.explicitA = True
        else:
            self.explicitA = False
            
    def solve(self):
        # initial state
        x = self.x0.copy()
        stepsize = self.stepsize
        
        k, flag = 0, 0
        ######## To remove
        l = 2
        s = 2
        l_k = 2*l + 1
        h_ = np.ones((l_k, l_k))/(l_k)**2
        kernel = np.zeros((64, 64))
        kernel[0:l_k,0:l_k] = np.ones((l_k, l_k))/(l_k)**2
        centered_kernel = np.zeros(kernel.shape)
        centered_kernel[0:s+1,0:s+1] = kernel[s:2*s+1, s:2*s+1]
        centered_kernel[0:s+1,-s:] = kernel[s:2*s+1, 0:s]
        centered_kernel[-s:,0:s+1] = kernel[0:s, s:2*s+1]
        centered_kernel[-s:,-s:] = kernel[0:s, 0:s]
        
        h_fft = np.fft.fft2(centered_kernel)
        hc_fft = np.conj(h_fft)
        A = lambda x : np.real(np.fft.ifft2(np.fft.fft2(x)*h_fft))
        AT = lambda x : np.real(np.fft.ifft2(np.fft.fft2(x)*hc_fft))

        #####################""
        
        while (k < self.maxit) and (flag == 0):
            x_old = x.copy()
            k += 1
    
            if self.explicitA:
                grad = self.A.T@(self.A @ x_old - self.b)
            else:
                grad = self.A(self.A(x_old, 1) - self.b, 2)
               
            x_new = self.proximal(x_old-stepsize*grad, stepsize)
            
            if self.adaptive:
                x_new = x_new + ((k-1)/(k+2))*(x_new - x_old)
            
            if LA.norm(x_new-x_old) <= self.abstol:
                flag = 1
              
            x = x_new.copy()
                
        return x, k


class RegularizedLinearRTO(Sampler):

    def __init__(self, target, x0 = None, maxit = 100, stepsize = 1e0, abstol=1e-10, **kwargs):

        super().__init__(target, x0 = x0, **kwargs)

        # Check target type
        if not isinstance(target, cuqi.distribution.Posterior):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")       

        # Check Linear model and Gaussian prior+likelihood
        if not isinstance(self.model, cuqi.model.LinearModel):
            raise TypeError("Model needs to be linear")

        if not hasattr(self.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        if not hasattr(self.prior.get_explicit_Gaussian(), "sqrtprec"):
            raise TypeError("prior must contain a sqrtprec attribute")

        if not hasattr(self.prior.get_explicit_Gaussian(), "sqrtprecTimesMean"):
            raise TypeError("Prior must contain a sqrtprecTimesMean attribute")

        if not callable(self.prior.get_proximal()):
            raise TypeError("Projector needs to be callable")

        # Modify initial guess        
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.prior.get_explicit_Gaussian().dim)

        # Other parameters
        self.maxit = maxit
        self.stepsize = stepsize
        self.abstol = abstol    
        self.proximal = self.prior.get_proximal()
                
        L1 = self.likelihood.distribution.sqrtprec
        L2 = self.prior.get_explicit_Gaussian().sqrtprec
        L2mu = self.prior.get_explicit_Gaussian().sqrtprecTimesMean

        # pre-computations
        self.m = len(self.data)
        self.n = len(self.x0)
        self.b_tild = np.hstack([L1@self.data, L2mu]) 

        if not callable(self.model):
            self.M = sp.sparse.vstack([L1@self.model, L2])
        else:
            # in this case, model is a function doing forward and backward operations
            def M(x, flag):
                if flag == 1:
                    out1 = L1 @ self.model.forward(x)
                    out2 = L2 @ x
                    out  = np.hstack([out1, out2])
                elif flag == 2:
                    idx = int(self.m)
                    out1 = self.model.adjoint(L1.T@x[:idx])
                    out2 = L2.T @ x[idx:]
                    out  = out1 + out2                
                return out   
            self.M = M       

    @property
    def prior(self):
        return self.target.prior

    @property
    def likelihood(self):
        return self.target.likelihood

    @property
    def model(self):
        return self.target.model     
    
    @property
    def data(self):
        return self.target.data

    def _sample(self, N, Nb):   
        Ns = N + Nb   # number of simulations        
        samples = np.empty((self.n, Ns))
                     
        # initial state   
        samples[:, 0] = self.x0
        for s in range(Ns-1):
            y = self.b_tild + np.random.randn(len(self.b_tild))
            sim = FISTA(self.M, y, samples[:, s], self.proximal,
                        maxit = self.maxit, stepsize = self.stepsize, abstol = self.abstol)         
            samples[:, s+1], _ = sim.solve()
            
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)
        # remove burn-in
        samples = samples[:, Nb:]
        
        return samples, None, None

    def _sample_adapt(self, N, Nb):
        return self._sample(N,Nb)




class ImplicitRegularizedGaussian(Distribution):

    def __init__(self, gaussian, proximal = None, projector = None, constraint = None, **kwargs):
        if (proximal is not None) + (projector is not None) + (constraint is not None) != 1:
            raise ValueError("Your life is failed")
        # Init from abstract distribution class
        
        # Geometry here is UGLY
        super().__init__(geometry = gaussian.geometry, **kwargs) 

        # Init specific to this distribution
        
        # Error checking if proximal is callable
        # Error checking if gaussian is actually a proximal
        
        self._gaussian = gaussian
        
        if proximal is not None:
            self._proximal = proximal
        elif projector is not None:
            self._proximal = lambda z, gamma: projector(z)
        elif (isinstance(constraint, str) and constraint.lower() in ["nonnegative", "nn"]):
            self._proximal = lambda z, gamma: ProjectNonnegative(z)
        elif (isinstance(constraint, str) and constraint.lower() in ["box"]):
            self._proximal = lambda z, gamma: ProjectBox(z)
        else:
            raise ValueError("Regularization not supported")
            

    def get_explicit_Gaussian(self):
        return self._gaussian
    
    def get_proximal(self):
        return self._proximal

    def logpdf(self, x):
        raise ValueError(
            f"The logpdf of a implicit regularized Gaussian distribution need not be defined.")
        
    def _sample(self, N, rng=None):
        raise ValueError(
            f"There is no known way of efficiently sampling from a implicit regularized Gaussian distribution need not be defined.")
            



#%% ADMM 


class ADMM(object):
    
    def __init__(self, A, b, penalties, rho, x0, maxit, innermaxit, innertol):
        self.A = A
        self.b = b
        self.penalties = penalties
        self.rho = rho
        self.x0 = x0
        self.maxit = maxit
        self.innermaxit = innermaxit
        self.innertol = innertol
            
    def solve(self):
        sqrt2rho = np.sqrt(2/self.rho)

        p = len(self.penalties)
        
        if not callable(self.A) and np.all([not callable(penalty.transformation) for penalty in self.penalties]):
            big_matrix = sp.sparse.vstack([sqrt2rho*self.A] + [penalty.transformation.get_matrix() if isinstance(penalty.transformation, Operator) else penalty.transformation for penalty in self.penalties])
        else:
            def big_matrix(x, flag):
                if flag == 1:
                    if not callable(self.A):
                        outA = sqrt2rho*self.A@x
                    else:
                        outA = sqrt2rho*self.A(x, 1)
                    outL = [penalty.transformation@x for penalty in self.penalties]
                    out  = np.hstack([outA] + outL)
                elif flag == 2:
                    if not callable(self.A):
                        cum = self.A.shape[0]
                        out = sqrt2rho*self.A.T@x[:cum]
                    else:
                        cum = self.A(x,3)
                        out = sqrt2rho*self.A(x[:cum],2)
                    for penalty in self.penalties:
                        out += penalty.transformation.T@x[cum:cum + penalty.transformation.shape[0]]   
                        cum += penalty.transformation.shape[0]
                return out       
        
        y_new = [np.zeros(penalty.transformation.shape[0]) for penalty in self.penalties]
        y_old = [np.zeros(penalty.transformation.shape[0]) for penalty in self.penalties]

        u_new = [np.zeros(penalty.transformation.shape[0]) for penalty in self.penalties]
        u_old = [np.zeros(penalty.transformation.shape[0]) for penalty in self.penalties]

        x_old = self.x0

        k = 0
        while k < self.maxit:        
            k += 1

            # Main update (Least Squares)
            big_vector = np.hstack([sqrt2rho*self.b] + [y_old[i] - u_old[i] for i in range(p)])
            solver = CGLS(big_matrix, big_vector, x_old, self.innermaxit, self.innertol, 0)
            x_new, _ = solver.solve()

            if k == self.maxit:
                return x_new

            # Regularization update
            for i, penalty in enumerate(self.penalties):
                y_new[i] = penalty.untransformed_proximal(penalty.transformation@x_new + u_old[i], self.rho)

            # Dual update
            for i, penalty in enumerate(self.penalties):
                u_new[i] = u_old[i] + (penalty.transformation@x_new - y_new[i])

            x_old, u_old, y_old = x_new, u_new, y_new
