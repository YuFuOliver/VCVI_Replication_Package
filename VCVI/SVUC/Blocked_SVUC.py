# Yu Fu
# Blocked VI for the SVUC model
# Note that this version can take both StanModel and custom log posterior (as a class) as input
# margins constructed for the SVUC model
# Change the YJ functionality to calculate computational time accurately (28 Oct 2024)
# use median of stable variational parameters to be consistent with reported ELBO, use spearman correlation

import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import bridgestan as bs
from scipy.stats import norm,skew,spearmanr
import math
pi = torch.tensor(math.pi)

from typing import NewType, Tuple, Callable, Optional
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

from VCVI.YJ import YJ

class Blocked_SVUC:

    @staticmethod
    def rep_trick(b: TensorWithGrad, nu: TensorWithGrad, L: TensorWithGrad, etat: TensorWithGrad, dim:int, isYJ:bool) -> TensorWithGrad:
        eps = norm.rvs(size = dim)
        eps = torch.from_numpy(eps)

        if isYJ:
            theta = b + nu**2 * ( YJ(etat).G( L @ eps ) )
        else:
            theta = b + nu**2 * ( L @ eps )

        return theta
    

    def __init__(self, d, isYJ, block_size, optimizer, sampling, 
                 stan_model: bs.StanModel, log_post: Optional[Callable] = None):

        if stan_model is None and log_post is None:
            raise ValueError("You must provide either a StanModel or a log posterior function.")
        
        self.model = stan_model
        self.log_post = log_post
        self.sampling = sampling
        self.I1 = torch.eye(d[0]) # identity matrix with block1
        self.I2 = torch.eye(d[1]) # identity matrix with block2
        self.I3 = torch.eye(d[2]) # identity matrix with block3
        self.isYJ = isYJ

        if self.log_post is not None:
            self.dim = log_post.dim
        else:
            self.dim = self.model.param_num()  # Set dimensionality from StanModel
        
        self.num_group = d.size
        self.d = d

        # Initialize variational parameters

        self.b = torch.zeros(self.dim, requires_grad=True)
        self.nu = torch.full((self.dim,), 0.5, requires_grad=True)
        self.l12 = torch.zeros(d[0]-1, requires_grad = True)
        self.l22 = torch.zeros(d[1]-1, requires_grad = True)
        self.etat = torch.zeros(self.dim, requires_grad = True)
        self.L3 = torch.eye(self.dim - d[0] - d[1], requires_grad = True)

        vari_para = [self.b, self.nu, self.l12, self.l22]
        
        if isYJ:
            vari_para.append(self.etat)
        
        if block_size == 3:
            vari_para.append(self.L3)
        
        # Define the optimizer (Adadelta)
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr=0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)
                                            

    def sample(self) -> TensorWithGrad:
        # deal with the variational parameters
        dim = self.dim

        L1_inv = torch.diag(torch.ones(self.d[0])) + torch.diag(self.l12, -1)
        L1 = torch.linalg.solve_triangular(L1_inv, self.I1 , upper = False, unitriangular = True)

        L2_inv = torch.diag(torch.ones(self.d[1])) + torch.diag(self.l22, -1)
        L2 = torch.linalg.solve_triangular(L2_inv, self.I2 , upper = False, unitriangular = True)

        L3 = torch.eye(dim - self.d[0] - self.d[1]) + torch.tril(self.L3, -1)
        L = torch.block_diag(L1, L2, L3)

        # Reparameterization trick
        theta = Blocked_SVUC.rep_trick(self.b,self.nu,L,self.etat,self.dim,self.isYJ)

        return theta
    
    def logh(self, theta: TensorWithGrad) -> Tuple[torch.Tensor,torch.Tensor]:
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
        
        # ----------------- deal with the variational parameters --------------------------
        dim = self.dim

        with torch.no_grad():            

            L1_inv = torch.diag(torch.ones(self.d[0])) + torch.diag(self.l12, -1)
            L2_inv = torch.diag(torch.ones(self.d[1])) + torch.diag(self.l22, -1)
            L3 = torch.eye(dim - self.d[0] - self.d[1]) + torch.tril(self.L3, -1)
            # L3_inv = torch.inverse(L3)
            L3_inv = torch.linalg.solve_triangular(L3, self.I3, upper = False, unitriangular = True)
            L_inv = torch.block_diag(L1_inv, L2_inv, L3_inv) # Note that |L_inv| = 1 = |L|

            b_ = self.b.detach()
            nu_ = self.nu.detach()
            etat_ = self.etat.detach()
            L_inv_ = L_inv.detach()
        
        thetac = theta.detach().clone().requires_grad_(True)

        yj = YJ(etat_)

        if self.isYJ:
            xx = yj.iG( (thetac - b_)/nu_**2 )
        else:
            xx = (thetac - b_)/nu_**2
        
        x = L_inv_ @ xx
        
        standard_normal = Normal(0.0, 1.0)

        if self.isYJ:
            log_deter_YJ = torch.sum( torch.log( yj.diG_dtheta( (thetac - b_)/nu_**2 ) ) )
        else:
            log_deter_YJ = 0


        logJ = torch.sum( standard_normal.log_prob(x) ) + \
            log_deter_YJ + torch.sum( -2 * torch.log(nu_) )
        
        logq_v = logJ

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

        b_record = torch.zeros((1000,self.dim))
        nu_record = torch.zeros((1000,self.dim))
        l12_record = torch.zeros((1000,self.d[0]-1))
        l22_record = torch.zeros((1000,self.d[1]-1))
        L3_record = torch.zeros((1000,self.dim - self.d[0] - self.d[1],self.dim - self.d[0] - self.d[1]))
        etat_record = torch.zeros((1000,self.dim))

        for iter in range(num_iter):

            if iter >= num_iter - 1000:
                with torch.no_grad():
                    b_record[iter - (num_iter - 1000)] = self.b.detach().clone()
                    nu_record[iter - (num_iter - 1000)] = self.nu.detach().clone()
                    l12_record[iter - (num_iter - 1000)] = self.l12.detach().clone()
                    l22_record[iter - (num_iter - 1000)] = self.l22.detach().clone()
                    L3_record[iter - (num_iter - 1000)] = self.L3.detach().clone()
                    etat_record[iter - (num_iter - 1000)] = self.etat.detach().clone()
            
            # return ELBO (last step) and update variational parameters
            ELBO = self.train_step() # key step!!!

            with torch.no_grad():
                ELBO_store[iter] = ELBO

            # if iter % 100 == 0:
            #     print(f"Iteration {iter}: ELBO = {ELBO.item()}")

        with torch.no_grad():
            avg_b,_ = torch.median(b_record, dim=0)
            avg_nu,_ = torch.median(torch.abs(nu_record), dim=0)
            avg_l12,_ = torch.median(l12_record, dim=0)
            avg_l22,_ = torch.median(l22_record, dim=0)
            avg_L3,_ = torch.median(L3_record, dim=0)
            avg_etat,_ = torch.median(etat_record, dim=0)

            L1_inv = torch.diag(torch.ones(self.d[0])) + torch.diag(avg_l12, -1)
            L1 = torch.linalg.solve_triangular(L1_inv, self.I1 , upper = False, unitriangular = True)

            L2_inv = torch.diag(torch.ones(self.d[1])) + torch.diag(avg_l22, -1)
            L2 = torch.linalg.solve_triangular(L2_inv, self.I2 , upper = False, unitriangular = True)
            
            L = torch.block_diag(L1, L2, avg_L3)

        if self.sampling:
            # sample from variational density
            print('Sampling from variational density...')
            sample_size = 100000
            theta_m = np.zeros((sample_size, self.dim))

            for i in range(sample_size):
                theta = Blocked_SVUC.rep_trick(avg_b,avg_nu,L,avg_etat,self.dim,self.isYJ)
                theta_m[i,:] = theta.numpy()
            
            vi_mean = np.mean(theta_m,axis=0)
            vi_std = np.std(theta_m,axis=0)
            vi_corr,_ = spearmanr(theta_m, axis=0)
            vi_skew = skew(theta_m, axis=0)

            return ELBO_store, vi_mean, vi_std, vi_corr, vi_skew
        
        else:
            return ELBO_store