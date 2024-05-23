# -*- coding: utf-8 -*-
#  Copyright 2024 Technical University of Denmark
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Authored by: Amal Alghamdi (DTU)
#  Reviewed by: Jakob Sauer Jørgensen (DTU)

from cuqi.distribution import UserDefinedDistribution,\
    Gaussian, GMRF
import numpy as np
from cuqi.samples import Samples


class MyDistribution(UserDefinedDistribution):
    """A custom distribution that is a combination of multiple Gaussian-type
    distributions (e.g. Gaussian and GMRF). The resulting distribution is of
    dimension equal to the sum of the dimensions of the individual
    distributions.

    Parameters
    ----------
    distribution_list : list
        A list of distributions that are to be combined.
    """ 
    def __init__(self, distribution_list):
        # Assert all distributions are Gaussian or GMRF distributions
        for d in distribution_list:
            assert isinstance(d, (Gaussian, GMRF))

        # Save the list of distributions as an attribute
        self.distribution_list = distribution_list

        # Call the constructor of the parent class
        super().__init__(dim=self.dim, logpdf_func=self.logpdf,
                         sample_func=self.sample)
    
    @property
    def dim(self):
        """Return the dimension of the distribution which is the sum of the
        dimensions of the individual distributions."""
        return np.sum([d.dim for d in self.distribution_list])
    
    def sample(self, n):
        """Return a sample of the distribution by concatenating samples from
        the individual distributions."""
        samples = np.concatenate(
            [d.sample(n).samples for d in self.distribution_list], axis=0)
        return Samples(samples, geometry=self.geometry)

    def logpdf(self, x):
        """Return the log of the probability density function of the
        distribution, which is the sum of the logpdf of the individual
        distributions."""
        start_index = 0
        logpdf = 0
        for i, d in enumerate(self.distribution_list):
            logpdf += d.logpdf(x[start_index:start_index+d.dim])
            start_index += d.dim
        return logpdf
    
    @property
    def sqrtprec(self):
        """Return the square root of the precision matrix of the distribution,
        which is the block diagonal matrix of the square root of the precision
        matrices of the individual distributions."""

        # Initialize and populate the block diagonal matrix of the square root
        # of the precision matrix
        sqrtprec = np.zeros((self.dim, self.dim))
        start_idx = 0
        for d in self.distribution_list:
            sqrtprec[start_idx:start_idx+d.dim, start_idx:start_idx+d.dim] =\
                d.sqrtprec.toarray()
            start_idx += d.dim
        return sqrtprec
    
    @property
    def sqrtprecTimesMean(self):
        """Return the square root of the precision matrix times the mean of the
        distribution, which is the concatenation of the square root of the
        precision matrix times the mean of the individual distributions. This
        is required for the linear RTO sampler."""
        sqrtprecTimesMean = np.zeros(self.dim)
        start_idx = 0
        for d in self.distribution_list:
            sqrtprecTimesMean[start_idx:start_idx+d.dim] = d.sqrtprecTimesMean
            start_idx += d.dim
        return sqrtprecTimesMean