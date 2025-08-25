# Yu Fu
# Mean Field Variational Inference (independent Gaussian)
# Note that this version can take both StanModel and custom log posterior (as a class) as input
# use median of stable variational parameters to be consistent with reported ELBO, use spearman correlation

import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import numpy as np
import bridgestan as bs
from scipy.stats import norm,skew,spearmanr
import math
pi = torch.tensor(math.pi)

from typing import NewType, Callable, Optional,Tuple
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

from .YJ import YJ

# ------------------------------ VI with factor Cov -----------------------------------------
class MFVI:

    # Note that log_post is an instance
    # logh is a method of this class, thus the name is retained

    @staticmethod
    def rep_trick(mu,d,dim) -> torch.Tensor:
        eps = norm.rvs(size = dim)
        eps = torch.from_numpy(eps)

        theta = mu + d * eps

        return theta

    def __init__(self, optimizer, sampling, 
                 stan_model: Optional[bs.StanModel], 
                 log_post: Optional[Callable] = None):
  
        if stan_model is None and log_post is None:
            raise ValueError("You must provide either a StanModel or a log posterior function.")
        
        self.model = stan_model
        self.sampling = sampling
        self.log_post = log_post
    
        if self.log_post is not None:
            self.dim = log_post.dim # dimension of this model
        else:
            self.dim = self.model.param_num()  # Set dimensionality from StanModel

        
        # Initialize variational parameters
        self.mu = torch.zeros(self.dim, requires_grad=True)
        self.d = torch.full((self.dim,), 0.5, requires_grad=True)

        vari_para = [self.mu, self.d]
        
        # Define the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr = 0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)
                                            

    def sample(self) -> TensorWithGrad:

        theta = MFVI.rep_trick(self.mu, self.d, self.dim)

        return theta
    
    def logh(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is not None:
            with torch.no_grad():
                theta_np = theta.detach().clone().numpy()
            
            logh_v, gr_logh = self.model.log_density_gradient(theta_np, propto=False)

            logh_v = torch.tensor(logh_v)
            gr_logh = torch.tensor(gr_logh)
            return logh_v, gr_logh
        else:
            logh_v, gr_logh = self.log_post.logh(theta)
            
            return logh_v, gr_logh

    def logq(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        # For the chain rule: dELBO/dtheta * dtheta/dlambda:
        # clone theta to thetac so that variational parameters only contribute to theta
        # detach variational parameters to avoid gradient tracking in logq computation
        
        m = self.dim
        mu = self.mu 
        d = self.d

        with torch.no_grad():
            mu_ = mu.detach()
            d_ = d.detach()
        
        thetac = theta.detach().clone().requires_grad_(True)

        z = thetac

        # compute the inverse of Sigma
        # Sigma_inv = torch.diag(1/d_**2)

        # compute logq
        log_Sigma_deter = torch.sum(torch.log(d_**2))

        diff = z - mu_
        Sigmainvdiff = 1/d_**2 * diff
        quad = torch.dot(diff, Sigmainvdiff) # (z - mu).T @ Sigma_inv @ (z - mu)
        logq_v = -0.5 * ( log_Sigma_deter + quad + m * torch.log(2*pi) )

        gr_logq = torch.autograd.grad(logq_v, thetac)[0]

        return logq_v.detach(), gr_logq

    
    def train_step(self) -> torch.Tensor:
        theta = self.sample()

        logh_v, gr_logh = self.logh(theta)
        logq_v, gr_logq = self.logq(theta)

        with torch.no_grad():
            ELBO = logh_v - logq_v
        
        gr_ELBO = gr_logh - gr_logq
        
        theta.backward(gr_ELBO)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        return ELBO


    def train(self, num_iter=10000):
        
        np.random.seed(33)
        
        ELBO_store= torch.zeros(num_iter)
        mu_store = torch.zeros(1000, self.dim)
        d_store = torch.zeros(1000, self.dim)

        for iter in range(num_iter):

            if iter >= num_iter - 1000:
                with torch.no_grad():
                    mu_store[iter - (num_iter - 1000)] = self.mu.clone().detach()
                    d_store[iter - (num_iter - 1000)] = self.d.clone().detach()
            
            # return ELBO (last step) and update variational parameters
            ELBO = self.train_step() # key step!!!

            with torch.no_grad():
                ELBO_store[iter] = ELBO

            if iter % 1000 == 0:
                print(f"Iteration {iter}: ELBO = {ELBO.item()}")

        with torch.no_grad():
            avg_mu, _ = torch.median(mu_store, dim=0)
            avg_d, _ = torch.median(torch.abs(d_store), dim=0)

        if self.sampling:

            # sample from variational density
            print('Sampling from variational density...')
            sample_size = 100000
            theta_m = np.zeros((sample_size, self.dim))

            for i in range(sample_size):
                theta = MFVI.rep_trick(avg_mu, avg_d, self.dim)
                theta_m[i,:] = theta.numpy()
            
            vi_mean = np.mean(theta_m,axis=0)
            vi_std = np.std(theta_m,axis=0)
            vi_corr, _ = spearmanr(theta_m, axis=0)
            vi_skew = skew(theta_m, axis=0)

            return ELBO_store, vi_mean, vi_std, vi_corr, vi_skew
        
        else:
            return ELBO_store