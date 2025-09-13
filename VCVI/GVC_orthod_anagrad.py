# Yu Fu
# Gaussian vector copula with an orthogonal correlation matrix
# Note that this version can take both StanModel and custom log posterior function as input
# margins are identity matrices
# this version is specifically designed for shrinkage priors such that dim = 2m+1. (horseshoe prior)
# Change the YJ functionality to calculate computational time accurately (10 Oct 2024) 
# a version of analytical gradient

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

# ------------------------- l(unconstraint) to Lam (-1 to 1) --------------------------------
def l2Lam(l):
    Lam_max = 1
    Lam_min = -1
    Lam = (Lam_max-Lam_min)/(torch.exp(-l)+1) + Lam_min

    return Lam


# ------------------------- Gaussian vector copula variational inference --------------------------------
class GVC_orthod_anagrad:

    def __init__(self, optimizer, sampling,
                 stan_model: Optional[bs.StanModel], 
                 log_post: Optional[Callable] = None):
        
        if stan_model is None and log_post is None:
            raise ValueError("You must provide either a StanModel or a log posterior function.")
        
        self.model = stan_model
        self.log_post = log_post
        self.sampling = sampling
    
        if self.log_post is not None:
            self.dim = log_post.dim
            self.dim1 = log_post.dim1
            self.dim2 = log_post.dim2
        else:
            self.dim = self.model.param_num()  # Set dimensionality from StanModel
            self.dim1 = int((self.dim - 1) / 2)
            self.dim2 = 2*self.dim1


        # Initialize variational parameters
        self.b = torch.zeros(self.dim)
        self.l = torch.zeros(self.dim1)
        self.s = torch.full((self.dim,), 0.5)

        vari_para = [self.b, self.l, self.s]
        
        # Define the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr=0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)


    def logh(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def logq(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # For the chain rule: dELBO/dtheta * dtheta/dlambda:
        # clone theta to thetac so that variational parameters only contribute to theta
        # detach variational parameters to avoid gradient tracking in logq computation

        Lam = l2Lam(self.l)
 
        eps1 = torch.from_numpy(norm.rvs(size = self.dim1))
        eps2 = torch.from_numpy(norm.rvs(size = self.dim1))
        eps3 = torch.from_numpy(norm.rvs(size = 1))
        eps = torch.cat([eps1,eps2,eps3], dim = 0)
        
        z1 = eps1
        z2 = Lam * eps1 + torch.sqrt(1 - Lam**2) * eps2
        z3 = eps3
        z = torch.cat([z1,z2,z3], dim = 0)

        theta = self.b + self.s**2 * z

        def Sigma_inverse(Lam):

            block1 = 1 + ( Lam**2 ) / ( 1 - Lam**2 )
            block2 = - Lam / ( 1 - Lam**2 )
            block3 = - Lam / ( 1 - Lam**2 )
            block4 = 1 / ( 1 - Lam**2 )

            return block1, block2, block3, block4
        

        block1,block2,block3,block4 = Sigma_inverse(Lam)

        # -------------------- value ----------------------------------------------
        # Sigma_deter = torch.prod( 1 - Lam_**2 )
        log_Sigma_deter = torch.sum( torch.log( 1 - Lam**2 ) )
        log_p = -0.5 * ( log_Sigma_deter + torch.sum(eps**2) + self.dim * torch.log(2*pi) )
        
        log_det = torch.sum( -2 * torch.log(self.s) )

        logq_v = log_p + log_det

        # -------------------- gradient ----------------------------------------------
        dlogq_dz1 =  - ( block1*z1 + block2*z2 )
        dlogq_dz2 =  - ( block3*z1 + block4*z2 )
        dlogq_dz3 =  - z3
        dlogq_dz = torch.cat([dlogq_dz1,dlogq_dz2,dlogq_dz3], dim = 0)

        gr_logq = 1/self.s**2 * dlogq_dz

        # ------------------- gradient for the rep_trick ------------------------------
        def dtheta_dl_eff(s,eps1,eps2,Lam,l,dim1,dim2):
            dtheta_dz = s**2
            dz2_dLam = eps1 - Lam / torch.sqrt( 1 - Lam**2 ) * eps2
            dLam_dl = ( 2 * torch.exp(-l) ) / ( (torch.exp(-l) + 1)**2 )

            dtheta_dl_v = dtheta_dz[dim1:dim2] * dz2_dLam * dLam_dl

            return dtheta_dl_v
        
        dtheta_dl_v = dtheta_dl_eff(self.s,eps1,eps2,Lam,self.l,self.dim1,self.dim2)
    

        return logq_v, gr_logq, z, theta, dtheta_dl_v

    
    def train_step(self) -> torch.Tensor:

        logq_v, gr_logq, z, theta, dtheta_dl_v = self.logq()
        logh_v, gr_logh = self.logh(theta)

        ELBO = logh_v - logq_v
        
        # gradients
        gr_ELBO = gr_logh - gr_logq
        
        self.b.grad =  gr_ELBO
        self.s.grad =  2*self.s*z * gr_ELBO
        self.l.grad = dtheta_dl_v * gr_ELBO[self.dim1:self.dim2]
        
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