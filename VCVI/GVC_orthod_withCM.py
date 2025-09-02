# Yu Fu
# Gaussian vector copula with an orthogonal correlation matrix, with a factor matrix as CM transform
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
pi = torch.tensor(math.pi)

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
class GVC_orthod_withCM:

    @staticmethod
    def rep_trick(b,l,C1,C2,d1,d2,d3,etat,dim1,isYJ) -> torch.Tensor:
        Lam = l2Lam(l)
        
        eps1 = torch.from_numpy(norm.rvs(size = dim1))
        eps2 = torch.from_numpy(norm.rvs(size = dim1))
        eps3 = torch.from_numpy(norm.rvs(size = 1))

        z1 = eps1
        z2 = Lam * eps1 + torch.sqrt(1 - Lam**2) * eps2
        z3 = eps3
        z = torch.cat([z1,z2,z3], dim = 0)

        F1 = C1 @ C1.T + torch.diag(d1**2)
        F2 = C2 @ C2.T + torch.diag(d2**2)
        F3 = d3**2

        if isYJ:
            theta = b + YJ(etat).G( torch.block_diag(F1,F2,F3) @ z )
        else:
            theta = b + torch.block_diag(F1,F2,F3) @ z

        return theta

    def __init__(self, p, isYJ, optimizer, sampling,
                 stan_model: Optional[bs.StanModel], 
                 log_post: Optional[Callable] = None):
        
        if stan_model is None and log_post is None:
            raise ValueError("You must provide either a StanModel or a log posterior function.")
        
        self.p = p
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
        self.b = torch.zeros(self.dim, requires_grad=True)
        self.l = torch.zeros(self.dim1, requires_grad=True)
        self.etat = torch.zeros(self.dim, requires_grad = True) # unconstraint eta
        self.C1t = torch.full((self.dim1,p), 0.001, requires_grad = True) # C = torch.tril(self.Ct)
        self.C2t = torch.full((self.dim1,p), 0.001, requires_grad = True) # C = torch.tril(self.Ct)
        self.d1 = torch.full((self.dim1,), 0.5, requires_grad=True)
        self.d2 = torch.full((self.dim1,), 0.5, requires_grad=True)
        self.d3 = torch.full((1,), 0.5, requires_grad=True)

        vari_para = [self.b, self.l, self.C1t, self.C2t, self.d1, self.d2, self.d3]

        if isYJ:
            vari_para.append(self.etat)
        
        # Define the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr=0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)
                                            

    def sample(self) -> TensorWithGrad:
        C1 = torch.tril(self.C1t)
        C2 = torch.tril(self.C2t)
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3

        theta = GVC_orthod_withCM.rep_trick(self.b, self.l, C1, C2, d1, d2, d3, self.etat, self.dim1, self.isYJ)
        
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

        def Sigma_inverse(Lam,dim1):
            block1 = 1 + ( Lam**2 ) / ( 1 - Lam**2 )
            block2 = - Lam / ( 1 - Lam**2 )
            block3 = - Lam / ( 1 - Lam**2 )
            block4 = 1 / ( 1 - Lam**2 )

            return block1,block2,block3,block4
        

        with torch.no_grad():
            b_ = self.b.detach()
            Lam_ = l2Lam(self.l).detach()
            C1_ = torch.tril(self.C1t).detach()
            C2_ = torch.tril(self.C2t).detach()
            d1_ = self.d1.detach()
            d2_ = self.d2.detach()
            d3_ = self.d3.detach()
            etat_ = self.etat.detach()

        thetac = theta.detach().clone().requires_grad_(True)

        block1,block2,block3,block4 = Sigma_inverse(Lam_,self.dim1)
        # Sigma_deter = torch.prod( 1 - Lam_**2 )
        log_Sigma_deter = torch.sum( torch.log( 1 - Lam_**2 ) )

        # ---------------- compute Finv and logdet(Finv) --------------------------
        def Finv_logdet(C_,d_):
            D2invC = torch.unsqueeze(1/d_**2,1) * C_
            inv_kernal = torch.eye(self.p) + C_.T @ D2invC

            # Finv = ( torch.diag(1/d_**2) 
            #             - D2invC @ torch.inverse( inv_kernal ) @ D2invC.T )

            Finv = torch.diag(1/d_**2) - D2invC @ torch.linalg.solve(inv_kernal, D2invC.T)

            log_F_deter = torch.sum(torch.log(d_**2)) + torch.logdet(inv_kernal)
            log_Finv_deter = - log_F_deter

            return Finv, log_Finv_deter

        F1inv, log_F1inv_deter = Finv_logdet(C1_,d1_)
        F2inv, log_F2inv_deter = Finv_logdet(C2_,d2_)
        F3inv = 1/d3_**2
        log_F3inv_deter = - torch.log(d3_**2)
        Finv = torch.block_diag(F1inv,F2inv,F3inv)
        log_Finv_deter = log_F1inv_deter + log_F2inv_deter + log_F3inv_deter

        # ------------------- inverse transform --------------------------
        if self.isYJ:
            z = Finv @ YJ(etat_).iG(thetac - b_)
        else:
            z = Finv @ (thetac - b_)

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
            log_deter_YJ = torch.sum( torch.log( YJ(etat_).diG_dtheta( thetac - b_ ) ) )
        else:
            log_deter_YJ = 0
        
        log_det = log_Finv_deter + log_deter_YJ

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

        ELBO_store= torch.zeros(num_iter)
        b_store = torch.zeros(1000, self.dim)
        l_store = torch.zeros(1000, self.dim1)
        C1_store = torch.zeros(1000, self.dim1, self.p)
        C2_store = torch.zeros(1000, self.dim1, self.p)
        d1_store = torch.zeros(1000, self.dim1)
        d2_store = torch.zeros(1000, self.dim1)
        d3_store = torch.zeros(1000, 1)
        etat_store = torch.zeros(1000, self.dim)

        for iter in range(num_iter):

            if iter >= num_iter - 1000:
                with torch.no_grad():
                    b_store[iter - (num_iter - 1000)] = self.b.clone().detach()
                    l_store[iter - (num_iter - 1000)] = self.l.clone().detach()
                    C1_store[iter - (num_iter - 1000),:,:] = torch.tril(self.C1t).clone().detach()
                    C2_store[iter - (num_iter - 1000),:,:] = torch.tril(self.C2t).clone().detach()
                    d1_store[iter - (num_iter - 1000)] = self.d1.clone().detach()
                    d2_store[iter - (num_iter - 1000)] = self.d2.clone().detach()
                    d3_store[iter - (num_iter - 1000)] = self.d3.clone().detach()
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
            avg_C1,_ = torch.median(C1_store, dim=0)
            avg_C2,_ = torch.median(C2_store, dim=0)
            avg_d1,_ = torch.median(torch.abs(d1_store), dim=0)
            avg_d2,_ = torch.median(torch.abs(d2_store), dim=0)
            avg_d3,_ = torch.median(torch.abs(d3_store), dim=0)
            avg_etat,_ = torch.median(etat_store, dim=0)

        if self.sampling:

            # sample from variational density
            print('Sampling from variational density...')
            sample_size = 100000
            theta_m = np.zeros((sample_size, self.dim))

            for i in range(sample_size):
                theta = GVC_orthod_withCM.rep_trick(avg_b, avg_l, avg_C1, avg_C2, avg_d1, avg_d2, avg_d3, avg_etat, self.dim1, self.isYJ)
                theta_m[i,:] = theta.numpy()
            
            vi_mean = np.mean(theta_m,axis=0)
            vi_std = np.std(theta_m,axis=0)
            vi_corr,_ = spearmanr(theta_m, axis=0)
            vi_skew = skew(theta_m, axis=0)

            return ELBO_store, vi_mean, vi_std, vi_corr, vi_skew
        
        else:
            return ELBO_store