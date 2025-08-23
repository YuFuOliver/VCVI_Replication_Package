# Yu Fu
# Gaussian vector copula with an orthogonal correlation matrix
# Note that this version can take both StanModel and custom log posterior function as input
# margins are identity matrices
# this version is specifically designed for shrinkage priors such that dim = 2m+1. (horseshoe prior)
# Change the YJ functionality to calculate computational time accurately (10 Oct 2024) 
# use median of stable variational parameters to be consistent with reported ELBO, use spearman correlation

import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import numpy as np
import bridgestan as bs
from scipy.stats import norm,skew,spearmanr
import math
pi = torch.tensor(math.pi).to('cuda')

from typing import NewType, Callable, Optional,Tuple
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

from .YJ import YJ

# ------------------------- l(unconstraint) to Lam (-1 to 1) --------------------------------
def l2Lam(l):
    Lam_max = 1
    Lam_min = -1
    Lam = (Lam_max-Lam_min)/(torch.exp(-l)+1) + Lam_min

    return Lam


# ------------------------- Gaussian vector copula variational inference --------------------------------
class GVC_orthod:

    @staticmethod
    def rep_trick(b,l,s,etat,dim1,isYJ) -> torch.Tensor:
        Lam = l2Lam(l)
        
        # Move random samples to the device
        eps1 = torch.from_numpy(norm.rvs(size = dim1)).to('cuda')
        eps2 = torch.from_numpy(norm.rvs(size = dim1)).to('cuda')
        eps3 = torch.from_numpy(norm.rvs(size = 1)).to('cuda')

        z1 = eps1
        z2 = Lam * eps1 + torch.sqrt(1 - Lam**2) * eps2
        z3 = eps3
        z = torch.cat([z1,z2,z3], dim = 0)

        if isYJ:
            theta = b + s**2 * ( YJ(etat).G( z ) )
        else:
            theta = b + s**2 * z

        return theta

    def __init__(self, isYJ, optimizer, sampling,
                 stan_model: Optional[bs.StanModel], 
                 log_post: Optional[Callable] = None):
        
        if stan_model is None and log_post is None:
            raise ValueError("You must provide either a StanModel or a log posterior function.")
        
        self.isYJ = isYJ
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
        self.b = torch.zeros(self.dim, requires_grad=True, device='cuda')
        self.l = torch.zeros(self.dim1, requires_grad=True, device='cuda')
        self.s = torch.full((self.dim,), 0.5, requires_grad=True, device='cuda')
        self.etat = torch.zeros(self.dim, requires_grad = True, device='cuda') # unconstraint eta

        vari_para = [self.b, self.l, self.s]

        if isYJ:
            vari_para.append(self.etat)
        
        # Define the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr=0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)
                                            

    def sample(self) -> TensorWithGrad:
        theta = GVC_orthod.rep_trick(self.b, self.l, self.s, self.etat, self.dim1, self.isYJ)
        
        return theta

    def logh(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is not None:
            with torch.no_grad():
                theta_np = theta.detach().clone().cpu().numpy()
            
            logh_v, gr_logh = self.model.log_density_gradient(theta_np, propto=False)

            logh_v = torch.tensor(logh_v).to('cuda')
            gr_logh = torch.tensor(gr_logh).to('cuda')
            return logh_v, gr_logh
        else:
            logh_v, gr_logh = self.log_post.logh(theta)
            
            return logh_v, gr_logh
    
    def logq(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        # For the chain rule: dELBO/dtheta * dtheta/dlambda:
        # clone theta to thetac so that variational parameters only contribute to theta
        # detach variational parameters to avoid gradient tracking in logq computation

        def Sigma_inverse(Lam,dim1):
            block1 = 1 + ( Lam**2 ) / ( 1 - Lam**2 )
            block2 = - Lam / ( 1 - Lam**2 )
            block3 = - Lam / ( 1 - Lam**2 )
            block4 = 1 / ( 1 - Lam**2 )

            return block1,block2,block3,block4
        

        with torch.no_grad():
            b_ = self.b.detach()
            Lam_ = l2Lam(self.l).detach()
            s_ = self.s.detach()
            etat_ = self.etat.detach()

        thetac = theta.detach().clone().requires_grad_(True)

        block1,block2,block3,block4 = Sigma_inverse(Lam_,self.dim1)
        # Sigma_deter = torch.prod( 1 - Lam_**2 )
        log_Sigma_deter = torch.sum( torch.log( 1 - Lam_**2 ) )

        if self.isYJ:
            z = YJ(etat_).iG( (thetac - b_) / s_**2 )
        else:
            z = (thetac - b_) / s_**2

        # -------------------- value ----------------------------------------------
        z1 = z[:self.dim1]
        z2 = z[self.dim1:self.dim2]
        z3 = z[self.dim2]

        Sigmainvz1 = block1*z1 + block2*z2
        Sigmainvz2 = block3*z1 + block4*z2
        Sigmainvz3 = z3
        Sigmainvz = torch.cat([Sigmainvz1,Sigmainvz2,Sigmainvz3.unsqueeze(0)], dim = 0)

        log_p = -0.5 * ( log_Sigma_deter + torch.dot(z, Sigmainvz) + self.dim * torch.log(2*pi) )

        if self.isYJ:
            log_deter_YJ = torch.sum( torch.log( YJ(etat_).diG_dtheta( (thetac - b_)/s_**2 ) ) )
        else:
            log_deter_YJ = 0
        
        log_det = torch.sum( -2 * torch.log(s_) ) + log_deter_YJ

        logq_v = log_p + log_det

        # -------------------- gradient ----------------------------------------------
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

        ELBO_store= torch.zeros(num_iter).to('cuda')
        b_store = torch.zeros(1000, self.dim).to('cuda')
        l_store = torch.zeros(1000, self.dim1).to('cuda')
        s_store = torch.zeros(1000, self.dim).to('cuda')
        etat_store = torch.zeros(1000, self.dim).to('cuda')

        for iter in range(num_iter):

            if iter >= num_iter - 1000:
                with torch.no_grad():
                    b_store[iter - (num_iter - 1000)] = self.b.clone().detach()
                    l_store[iter - (num_iter - 1000)] = self.l.clone().detach()
                    s_store[iter - (num_iter - 1000)] = self.s.clone().detach()
                    etat_store[iter - (num_iter - 1000)] = self.etat.clone().detach()
            
            # return ELBO (last step) and update variational parameters
            ELBO = self.train_step() # key step!!!

            with torch.no_grad():
                ELBO_store[iter] = ELBO

            if iter % 1000 == 0:
                print(f"Iteration {iter}: ELBO = {ELBO.item()}")

        with torch.no_grad():
            avg_b,_ = torch.median(b_store, dim=0)
            avg_l,_ = torch.median(l_store, dim=0)
            # avg_s = torch.mean(s_store, dim=0)
            avg_s,_ = torch.median(torch.abs(s_store), dim=0)
            avg_etat,_ = torch.median(etat_store, dim=0)

        if self.sampling:

            # sample from variational density
            print('Sampling from variational density...')
            sample_size = 100000
            theta_m = np.zeros((sample_size, self.dim))

            for i in range(sample_size):
                theta = GVC_orthod.rep_trick(avg_b, avg_l, avg_s, avg_etat, self.dim1, self.isYJ)
                theta_m[i,:] = theta.cpu().numpy()
            
            vi_mean = np.mean(theta_m,axis=0)
            vi_std = np.std(theta_m,axis=0)
            vi_corr,_ = spearmanr(theta_m, axis=0)
            vi_skew = skew(theta_m, axis=0)

            return ELBO_store, vi_mean, vi_std, vi_corr, vi_skew
        
        else:
            return ELBO_store