# Yu Fu
# Mean Field Variational Inference
# Note that this version can take both StanModel and custom log posterior (as a class) as input

import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import numpy as np
import bridgestan as bs
from scipy.stats import norm,skew
import math
pi = torch.tensor(math.pi)

from typing import NewType, Callable, Optional,Tuple
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

# ------------------------------ VI with factor Cov -----------------------------------------
class MFVI_anagrad:

    # Note that log_post is an instance
    # logh is a method of this class, thus the name is retained


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
        self.mu = torch.zeros(self.dim)
        self.d = torch.full((self.dim,), 0.5)

        vari_para = [self.mu, self.d]
        
        # Define the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr = 0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)

    
    def logh(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model is not None:
            with torch.no_grad():
                theta_np = theta.detach().clone().numpy()
            
            logh_v, gr_logh = self.model.log_density_gradient(theta_np, propto=False)

            logh_v = torch.tensor(logh_v)
            gr_logh = torch.tensor(gr_logh)
            return logh_v, gr_logh
        else:
            theta_autodiff = theta.clone().requires_grad_(True)
            logh_v, gr_logh = self.log_post.logh(theta_autodiff)
            
            return logh_v, gr_logh

    def logq(self) -> Tuple[torch.Tensor, torch.Tensor]:
        
        eps = norm.rvs(size = self.dim)
        eps = torch.from_numpy(eps)

        z = self.mu + self.d * eps

        # compute the inverse of Sigma
        # Sigma_inv = torch.diag(1/self.d**2)

        # compute logq
        log_Sigma_deter = torch.sum(torch.log(self.d**2))

        diff = z - self.mu
        Sigmainvdiff = 1/self.d**2 * diff
        quad = torch.dot(diff, Sigmainvdiff) # (z - mu).T @ Sigma_inv @ (z - mu)
        logq_v = -0.5 * ( log_Sigma_deter + quad + self.dim * torch.log(2*pi) )

        gr_logq = - Sigmainvdiff

        return logq_v, gr_logq, z, eps

    
    def train_step(self) -> torch.Tensor:

        logq_v, gr_logq, theta, eps = self.logq()
        logh_v, gr_logh = self.logh(theta)


        ELBO = logh_v - logq_v
        
        gr_ELBO = gr_logh - gr_logq
        
        self.mu.grad = gr_ELBO
        self.d.grad = eps * gr_ELBO
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        return ELBO


    def train(self, num_iter=10000):
        
        np.random.seed(33)
        
        ELBO_store= torch.zeros(num_iter)

        for iter in range(num_iter):
            
            # return ELBO (last step) and update variational parameters
            ELBO = self.train_step() # key step!!!
            ELBO_store[iter] = ELBO

            # if iter % 100 == 0:
            #     print(f"Iteration {iter}: ELBO = {ELBO.item()}")

        if self.sampling:
            pass
        else:
            return ELBO_store