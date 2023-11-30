import numpy as np
import pywt
from scipy.stats import gennorm, norm
import matplotlib.pyplot as plt
# Class that defines the Besov prior with all its functionalities
class besov_prior:
    def __init__(self,J,delt,level, s=1, p=1, wavelet="db1"):
        # s paramter
        self.s = s
        # p parameter
        self.p = p
        # wavelet basis
        self.wavelet = wavelet
        # Maximum amount of wavelet levels, that is dimension n=2^J.
        self.J = J
        # Prior regularization paramter
        self.delt = delt
        # Level determines the amount of levels to not use in the Discrete wavelet transform
        self.level = level
 
    def inverse_weights2(self):
        # inverse_weights2 computes in the inverse weights of the Besov prior using the natural wavelet weights.
        # These weights is the one we will base the remaining methods on.
        # Dimension
        n = 2**self.J
        # Allocation
        inv_weights = np.zeros(n)
        # Computing scaling function weigth
        inv_weights[0] = 1
        # Computing the weights for each wavelet level.
        for i in range(self.J):
            inv_weights[2**i:2**(i+1)]=2**(-i*(self.s+0.5-1/self.p))
        #return np.sqrt(n)*inv_weights 
        return inv_weights
    
    def inv_wavelet_weigth(self, signal):
        # Computing B^-1 which is the inverse wavelet transform of the inverse Besov scaled signal.
        # Computing the inverse Besov weights
        inv_weigth = self.inverse_weights2()
        # Normalizing and multiplying unto the signal
        coeff = inv_weigth*signal/np.linalg.norm(inv_weigth,ord=2)
        # Setting up the list that contains the wavelet coefficients
        List = []
        # Setup the coefficients for inverse wavelet transform
        # Computing the scaling function coefficients which is not scaled.
        #List.append(np.sqrt(len(signal))*signal[0:2**(self.level)])
        List.append(signal[0:2**(self.level)]/np.linalg.norm(inv_weigth,ord=2))
        # Computing the wavelet coefficient at each level above self.level
        for j in range(self.level,self.J):
         List.append(coeff[2 ** j:2 ** (j + 1)])
        # Doing the inverse wavelet transform on the wavelet coefficients
        return pywt.waverec(List, self.wavelet, mode='periodization')
    
    def prior_to_normal(self, x):
        # prior_to_normal computes the map g and its derivative that transforms a p-norm prior to a standard Gaussian.
        # Allocation
        g = np.zeros(len(x))
        g_diff = np.zeros(len(x))
        # Splitting the indicies in two categories. Index 1 is where we have to approximate the map. Index 2 is where we can compute the map directly.
        #index1 = np.logical_or(x<-7, x>7)
        #index2 = np.logical_and(x>=-7, x<=7)
        # Scale parameter of the p-gaussian distribution
        scal = (self.p/(self.delt))**(1/self.p)
        # Regularization paramter delt/p
        #lam = self.delt/self.p
        # Computing the cdf of the standard normal distribution
        index1 = x > 0
        index2 = x <= 0
        cdf1 = norm.cdf(-x[index1])
        cdf2 = norm.cdf(x[index2])
        g[index1] = -gennorm.ppf(cdf1, self.p, loc=0, scale=scal)
        g[index2] = gennorm.ppf(cdf2, self.p, loc=0, scale=scal)
        g_diff[index1] = norm.pdf(x[index1]) / (gennorm.pdf(g[index1], self.p, loc=0, scale=scal))
        g_diff[index2] = norm.pdf(-x[index2]) / (gennorm.pdf(g[index2], self.p, loc=0, scale=scal))
        #cdf = norm.cdf(x[index2])
        # computing the inverse cdf of the p-gaussian to the standard normal cdf (which is precise map g)
        #g[index2] = gennorm.ppf(cdf, self.p, scale=scal)
        # Computing the derivative of the transformation map
        #g_diff[index2] = norm.pdf(x[index2]) / (gennorm.pdf(g[index2], self.p, scale=scal))
        # Computing the approximation for indicies where we do not have enough numerical precision
        #g[index1] = np.sign(x[index1]) * (np.abs(x[index1]) ** 2 / (2 * lam)) ** (1 / self.p)
        #g_diff[index1] = x[index1]/(lam*self.p)*(np.abs(x[index1])**2/(2*lam))**(1/self.p-1)
        # Returning the computed results
        return g, g_diff

    def transform(self,x):
        # Transform computes the full prior transformation B^1g(*)
        # Computing g
        g = self.prior_to_normal(x)[0]
        # Computing B^-1*g
        return self.inv_wavelet_weigth(g)
       # return self.inv_wavelet_weigth(x)

    def jac_const(self,N):
        # Computing the Besov matrix B^-1 which is used as the prior jacobian.
        I = np.identity(N)
        A = np.zeros((N,N))
        # Evaluating the linear transform on the identity matrix provides the matrix B^-1.
        for i in range(N):
            A[:,i] = self.inv_wavelet_weigth(I[:,i])  
        return A    

    # Extra functions which is currently not used.
    def weigths(self):
        n = 2**self.J
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = (i+1)**(self.s+0.5-1/self.p)
        return 1/np.sqrt(n)*weights
    
    def inverse_weigths(self):
        # Inverse weigths computes the inverse of the first version of the Besov weights
        # Dimension
        n = 2**self.J
        # Allocation
        inv_weights = np.zeros(n)
        # Computing the weights
        for i in range(n):
            inv_weights[i]=(i+1)**(-(self.s+0.5-1/self.p))
        return inv_weights

    def weights2(self):
        # inverse_weights2 computes in the inverse weights of the Besov prior using the natural wavelet weights.
        # These weights is the one we will base the remaining methods on.
        # Dimension
        n = 2**self.J
        # Allocation
        weights = np.zeros(n)
        # Computing scaling function weigth
        weights[0] = 1
        # Computing the weights for each wavelet level.
        for i in range(self.J):
            weights[2**i:2**(i+1)]=2**(i*(self.s+0.5-1/self.p))
        return 1/np.sqrt(n)*weights 

    def wavelet_weigth(self, signal):
         wavelet_coefficients = pywt.wavedec(signal, self.wavelet, mode='periodization', level=self.J)
         weights = self.weights2()
         weights[0:2**(self.level)] = 1/np.sqrt(2**self.J)
         return weights*pywt.coeffs_to_array(wavelet_coefficients)[0]

    def compute_besov_matrix(self):
        n = 2**self.J
        I = np.identity(n)
        B = np.zeros((n,n))
        for i in range(n):
            B[:,i]= self.wavelet_weigth(I[:,i])
        return B

    def norm(self, signal):
        return np.linalg.norm(self.wavelet_weigth(signal) , ord=self.p)**self.p

    def norm_inv_trans(self, signal):
        return np.linalg.norm(self.inv_wavelet_weigth(signal), ord=self.p)**self.p     

if __name__=="__main__":

    J = 10
    N = 2**J
    wavelet = 'db1'
    alphagrid = np.linspace(-5,5, N)
    xgrid = np.linspace(0,1,N)
    prior = besov_prior(J,2*32)
    beta = 1
    mu = 0
    lam = 32
    alpha = np.exp(np.log(1/lam)/beta)
    params = dict([('alpha',alpha), ('beta', beta), ('mu',mu),('lambda',lam)])
    g, gdiff = prior.besov_to_normal(alphagrid)
    Invmap=gennorm.ppf(xgrid, beta, loc=mu, scale=alpha)
    g_true = -1/lam*np.sign(alphagrid)*np.log(1-2*np.abs(norm.cdf(alphagrid)-0.5))
    gdiff_true = norm.pdf(alphagrid)/(lam*norm.cdf(-np.abs(alphagrid)))
    fig2, ax2=plt.subplots(2,1)
    ax2[0].plot(alphagrid,g)
    ax2[0].plot(alphagrid,g_true)
    ax2[1].plot(alphagrid,gdiff)
    ax2[1].plot(alphagrid,gdiff_true)
    print(np.sum(np.abs(g-g_true)),np.sum(np.abs(gdiff-gdiff_true)))
    plt.show()




        

