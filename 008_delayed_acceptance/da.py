# %% DA class
import cuqi
import numpy as np
from cuqi.sampler import ProposalBasedSampler
class DA(ProposalBasedSampler):
    """Delayed Acceptance sampler.

    Allows sampling of a target distribution by a two-stage stragy, each having its accept/reject step.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution` or lambda function
        The target distribution to sample. Custom logpdfs are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.

    reduced_target : `cuqi.distribution.Distribution` or lambda function
        The approxiamted target distribution, which is related to a reduce model. Custom logpdfs are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
        
    proposal : `cuqi.distribution.Distribution` or callable method
        The proposal to sample from. If a callable method it should provide a single independent sample from proposal distribution. Defaults to a Gaussian proposal.  *Optional*.

    scale : float
        Scale parameter used to define correlation between previous and proposed sample in random-walk.  *Optional*.

    x0 : ndarray
        Initial parameters. *Optional*

    dim : int
        Dimension of parameter space. Required if target and proposal are callable functions. *Optional*.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
    """
    def __init__(self, target, reduced_target, proposal=None, scale=None, x0=None, dim=None, **kwargs):
        """ Delayed Acceptance (DA) sampler. Default (if proposal is None) with proposal that is Gaussian with identity covariance"""
        super().__init__(target, proposal=proposal, scale=scale,  x0=x0, dim=dim, **kwargs)
        self.reduced_target = reduced_target
        self.second_stage_counter = 0
        self.second_stage_accepted_counter = 0


    @ProposalBasedSampler.proposal.setter 
    def proposal(self, value):
        fail_msg = "Proposal should be either None, symmetric cuqi.distribution.Distribution or a lambda function."

        if value is None:
            self._proposal = cuqi.distribution.Gaussian(np.zeros(self.dim), 1)
        elif not isinstance(value, cuqi.distribution.Distribution) and callable(value):
            raise NotImplementedError(fail_msg)
        elif isinstance(value, cuqi.distribution.Distribution) and value.is_symmetric:
            self._proposal = value
        else:
            raise ValueError(fail_msg)
        self._proposal.geometry = self.target.geometry

    def _sample(self, N, Nb):
        if self.scale is None:
            raise ValueError("Scale must be set to sample without adaptation. Consider using sample_adapt instead.")
        
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        target_eval = np.empty(Ns)
        reduced_target_eval = np.empty(Ns)
        acc = np.zeros(Ns, dtype=int)

        # initial state    
        samples[:, 0] = self.x0
        target_eval[0] = self.target.logd(self.x0)
        reduced_target_eval[0] = self.reduced_target.logd(self.x0)
        acc[0] = 1

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], reduced_target_eval[s+1], acc[s+1] = self.single_update(samples[:, s], target_eval[s], reduced_target_eval[s])
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)

        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, '\n')
        #
        return samples, target_eval, accave

    def _sample_adapt(self, N, Nb):
        # Set intial scale if not set
        if self.scale is None:
            self.scale = 0.1
            
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        target_eval = np.empty(Ns)
        reduced_target_eval = np.empty(Ns)
        acc = np.zeros(Ns)

        # initial state    
        samples[:, 0] = self.x0
        target_eval[0] = self.target.logd(self.x0)
        reduced_target_eval[0] = self.reduced_target.logd(self.x0)
        acc[0] = 1

        # initial adaptation params 
        Na = int(0.1*N)                              # iterations to adapt
        hat_acc = np.empty(int(np.floor(Ns/Na)))     # average acceptance rate of the chains
        lambd = self.scale
        star_acc = 0.234    # target acceptance rate RW
        i, idx = 0, 0

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], reduced_target_eval[s+1], acc[s+1] = self.single_update(samples[:, s], target_eval[s], reduced_target_eval[s])
            
            # adapt prop spread using acc of past samples
            if ((s+1) % Na == 0):
                # evaluate average acceptance rate
                hat_acc[i] = np.mean(acc[idx:idx+Na])

                # d. compute new scaling parameter
                zeta = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lambd = np.exp(np.log(lambd) + zeta*(hat_acc[i]-star_acc))

                # update parameters
                self.scale = min(lambd, 1)

                # update counters
                i += 1
                idx += Na

            # display iterations
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample


        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        reduced_target_eval = reduced_target_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, 'MCMC scale:', self.scale, '\n')
        
        # return samples, target_eval, reduced_target_eval, accave
        return samples, target_eval, accave


    def single_update(self, x_t, target_eval_t, reduced_target_eval_t):
        # propose state
        xi = self.proposal.sample(1)   # sample from the proposal
        x_star = x_t + self.scale*xi.flatten()   # MH proposal

        # evaluate reduced_target
        reduced_target_eval_star = self.reduced_target.logd(x_star)

        # ratio and acceptance probability
        reduced_ratio = reduced_target_eval_star - reduced_target_eval_t  # proposal is symmetric
        alpha = min(0, reduced_ratio)

        acc = 0

        # first stage: accept/reject with reduced target
        u_theta = np.log(np.random.rand())
        if (u_theta <= alpha):
            # second stage: accept/reject with reduced target
            self.second_stage_counter += 1
            target_eval_star = self.target.logd(x_star)
            ratio = target_eval_star - target_eval_t - reduced_ratio

            beta = min(0, ratio)
            v_theta = np.log(np.random.rand())
            if (v_theta <= beta):
                # print('accepted with full model')
                self.second_stage_accepted_counter += 1
                x_next = x_star
                reduced_target_eval_next = reduced_target_eval_star
                target_eval_next = target_eval_star
                acc = 1
            else:
                # print('rejected with full model')
                x_next = x_t
                target_eval_next = target_eval_t
                reduced_target_eval_next = reduced_target_eval_t
                acc = 0
        else:
            x_next = x_t
            target_eval_next = target_eval_t
            reduced_target_eval_next = reduced_target_eval_t
            acc = 0
        
        return x_next, target_eval_next, reduced_target_eval_next, acc
