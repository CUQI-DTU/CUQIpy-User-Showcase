import numpy as np
import numpy.random as random

def gen_fin_diff(n):
    A = np.eye(n);
    for i in range(1,n):
        A[i, i-1] = -1;
    return A[1:,:];

def gen_fin_diff_extra(n):
    A = np.eye(n);
    for i in range(1,n):
        A[i, i-1] = -1;
    return A;
    
def gen_fin_diff_2(n):
    A = -2*np.eye(n);
    for i in range(1,n):
        A[i, i-1] = 1;
        A[i-1, i] = 1;
    return A[1:n-1,:];
    

def create_problem(blur_parameter, lambd_noise, seed, n=128):
    x_space = np.linspace(0,1,n);
    x_true = np.zeros((n));
    
    def f(x):
        if 0.1 <= x and x <= 0.3:
            return 1;
        if 0.4 <= x and x <= 0.5:
            return x-0.4;
        if 0.5 <= x and x <= 0.6:
            return 0.1-(x-0.5);
        if 0.7 <= x and x <= 0.9:
            return 1.4*np.exp(-0.007*(128*(x-0.8))**2)-0.40;
        return 0;
    
    for i in range(n):
        x_true[i] = f(x_space[i]);
    
    # Gaussian blur matrix
    def gen_A(n, gamma):
        h = 1.0/n;
        A = np.zeros((n, n));
        for i in range(n):
            for j in range(n):
                A[i, j] = (h/(gamma*np.sqrt(2*np.pi)))*np.exp(-0.5*((h*(i-j))/gamma)**2);
        return A;
    
    # finite difference operator

    
    par = blur_parameter#0.02;
    
    # Generate data
    x2_space = np.linspace(0,1,2*n-1);
    x2_true = np.zeros((2*n-1));
    
    def f(x):
        if 0.1 <= x and x <= 0.3:
            return 1;
        if 0.4 <= x and x <= 0.5:
            return x-0.4;
        if 0.5 <= x and x <= 0.6:
            return 0.1-(x-0.5);
        if 0.7 <= x and x <= 0.9:
            return 1.4*np.exp(-0.007*(128*(x-0.8))**2)-0.40;
        return 0;
    
    for i in range(2*n-1):
        x2_true[i] = f(x2_space[i]);
        
    b = (gen_A(2*n-1, par)@x2_true)[0::2];
    
    A = gen_A(n, par);
    lambd = lambd_noise #1000;
    random.seed(seed)
    noise = (1.0/np.sqrt(lambd))*random.normal(size=n);
    b_noisy = b + noise;
    
    return (x_space, A, b_noisy, x2_true[0::2])