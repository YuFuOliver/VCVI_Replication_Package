# Yu Fu
# Gaussian vector copula with a factor correlation matrix
# Note that this version can take both StanModel and custom log posterior (as a class) as input
# margins are identity matrices
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

class GVC_factor:

    # Note that log_post is an instance
    # logh is a method of this class, thus the name is retained

    @staticmethod
    def rep_trick(b: TensorWithGrad, s: TensorWithGrad, Ct: TensorWithGrad, zetat: TensorWithGrad,
               etat: TensorWithGrad, k: np.ndarray, p: int, identities: list, isYJ) -> Tuple[TensorWithGrad, torch.Tensor]:

        dim = np.sum(k)

        zeta = zetat**2
        C = torch.tril(Ct)
        
        Omegat = C @ C.T + zeta * torch.eye(dim, device = 'cuda')
        
        # ----------------------- Compute A and A_inv -----------------------
        blocks_chol = []         # for A_inv
        blocks_chol_inv = []     # for A

        start = 0
        for j in range(len(k)):
            size = k[j]
            
            block = Omegat[start:start+size, start:start+size]

            block_chol = torch.linalg.cholesky(block)
            block_chol_inv = torch.linalg.solve_triangular(block_chol, identities[j], upper = False)

            blocks_chol.append(block_chol)
            blocks_chol_inv.append(block_chol_inv)

            start += size

        A_inv = torch.block_diag(*blocks_chol)
        A = torch.block_diag(*blocks_chol_inv)

        # ----------------------- rep trick -----------------------
        eps1 = torch.from_numpy(norm.rvs(size = dim)).to('cuda')
        eps2 = torch.from_numpy(norm.rvs(size = p)).to('cuda')

        z = torch.sqrt(zeta) * A @ eps1 + A @ C @ eps2

        if isYJ:
            theta = b + s**2 * ( YJ(etat).G( z ) )
        else:
            theta = b + s**2 * z

        return theta, A_inv.detach().clone()
    


    def __init__(self, k, p, isYJ, optimizer, sampling, 
                 stan_model: Optional[bs.StanModel], 
                 log_post: Optional[Callable] = None):
  
        if stan_model is None and log_post is None:
            raise ValueError("You must provide either a StanModel or a log posterior function.")
        
        self.isYJ = isYJ
        self.model = stan_model
        self.k = k
        self.p = p
        self.identities = [torch.eye(i, device='cuda') for i in k] # to solve linear system
        self.sampling = sampling
        self.log_post = log_post
    
        if self.log_post is not None:
            self.dim = log_post.dim # dimension of this model
        else:
            self.dim = self.model.param_num()  # Set dimensionality from StanModel

        
        # Initialize variational parameters
        self.b = torch.zeros(self.dim, requires_grad=True, device='cuda')
        self.s = torch.full((self.dim,), 0.5, requires_grad=True, device='cuda')
        self.Ct = torch.full((self.dim,p), 0.001, requires_grad = True, device='cuda') # C = torch.tril(self.Ct)
        self.zetat = torch.ones(1, requires_grad=True, device='cuda')
        self.etat = torch.zeros(self.dim, requires_grad = True, device='cuda') # unconstraint eta

        vari_para = [self.b, self.s, self.Ct, self.zetat]

        if isYJ:
            vari_para.append(self.etat)
        
        # Define the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr = 0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)
                                            

    def sample(self) -> Tuple[TensorWithGrad, torch.Tensor]:

        theta, A_inv = GVC_factor.rep_trick(self.b,self.s,self.Ct,self.zetat,self.etat,self.k,self.p,self.identities,self.isYJ)

        return theta, A_inv
    
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

    def logq(self, theta: TensorWithGrad, A_inv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = self.dim
        p = self.p
        
        with torch.no_grad():
            b_ = self.b.detach()
            s_ = self.s.detach()
            Ct_ = self.Ct.detach()
            zetat_ = self.zetat.detach()
            etat_ = self.etat.detach()

        zeta_ = zetat_**2
        C_ = torch.tril(Ct_)

        thetac = theta.detach().clone().requires_grad_(True)

        if self.isYJ:
            z = YJ(etat_).iG( (thetac - b_) / s_**2 )
        else:
            z = (thetac - b_) / s_**2
        
        # ------------ Compute the inverse and log_det of the covariance matrix --------
        # inv and logdet of Omegat
        zeta_vec = zeta_ * torch.ones(dim, device = 'cuda')
        
        D2invC = torch.unsqueeze(1/zeta_vec,1) * C_
        inv_kernal = torch.eye(p, device = 'cuda') + C_.T @ D2invC

        # Omegat_inv = ( torch.diag(1/zeta_vec) 
        #             - D2invC @ torch.inverse( inv_kernal ) @ D2invC.T )

        Omegat_inv = torch.diag(1/zeta_vec) - D2invC @ torch.linalg.solve(inv_kernal, D2invC.T)

        log_Omegat_deter = torch.sum(torch.log(zeta_vec)) + torch.logdet(inv_kernal)
        
        # inv and logdet of Omega
        Omega_inv = A_inv.T @ Omegat_inv @ A_inv

        log_Ainv_deter = torch.sum(torch.log(torch.diag(A_inv)))
        log_Omega_deter = - 2*log_Ainv_deter + log_Omegat_deter
        
        # compute logq
        if self.isYJ:
            log_deter_YJ = torch.sum( torch.log( YJ(etat_).diG_dtheta( (thetac - b_)/s_**2 ) ) )
        else:
            log_deter_YJ = 0
        
        log_det = torch.sum( -2 * torch.log(s_) ) + log_deter_YJ

        diff = z
        quad = torch.dot(diff, torch.matmul(Omega_inv, diff)) # (z - mu).T @ Sigma_inv @ (z - mu)
        logq_v = -0.5 * ( log_Omega_deter + quad + dim * torch.log(2*pi) ) + log_det

        gr_logq = torch.autograd.grad(logq_v, thetac)[0]

        return logq_v.detach(), gr_logq
    
    def train_step(self) -> torch.Tensor:
        theta, A_inv = self.sample()

        logh_v, gr_logh = self.logh(theta)
        logq_v, gr_logq = self.logq(theta, A_inv)

        with torch.no_grad():
            ELBO = logh_v - logq_v
        
        gr_ELBO = gr_logh - gr_logq
        
        theta.backward(gr_ELBO)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        return ELBO


    def train(self, num_iter=10000):
        
        np.random.seed(33)
        
        ELBO_store= torch.zeros(num_iter, device='cuda')
        b_store = torch.zeros(1000, self.dim, device='cuda')
        s_store = torch.zeros(1000, self.dim, device='cuda')
        Ct_store = torch.zeros(1000, self.dim, self.p, device='cuda')
        zetat_store = torch.zeros(1000, 1, device='cuda')
        etat_store = torch.zeros(1000, self.dim, device='cuda')

        for iter in range(num_iter):

            if iter >= num_iter - 1000:
                with torch.no_grad():
                    b_store[iter - (num_iter - 1000)] = self.b.clone().detach()
                    s_store[iter - (num_iter - 1000)] = self.s.clone().detach()
                    Ct_store[iter - (num_iter - 1000),:,:] = self.Ct.clone().detach()
                    zetat_store[iter - (num_iter - 1000)] = self.zetat.clone().detach()
                    etat_store[iter - (num_iter - 1000)] = self.etat.clone().detach()
            
            # return ELBO (last step) and update variational parameters
            ELBO = self.train_step() # key step!!!

            with torch.no_grad():
                ELBO_store[iter] = ELBO

            if iter % 1000 == 0:
                print(f"Iteration {iter}: ELBO = {ELBO.item()}")

        with torch.no_grad():
            avg_b,_ = torch.median(b_store, dim=0)
            avg_s,_ = torch.median(torch.abs(s_store), dim=0)
            avg_Ct,_ = torch.median(Ct_store, dim=0)
            avg_zetat,_ = torch.median(torch.abs(zetat_store), dim=0)
            avg_etat,_ = torch.median(etat_store, dim=0)

        if self.sampling:

            # sample from variational density
            print('Sampling from variational density...')
            sample_size = 100000
            theta_m = np.zeros((sample_size, self.dim))

            for i in range(sample_size):
                theta,_ = GVC_factor.rep_trick(avg_b,avg_s,avg_Ct,avg_zetat,avg_etat,self.k,self.p,self.identities,self.isYJ)
                theta_m[i,:] = theta.cpu().numpy()
            
            vi_mean = np.mean(theta_m,axis=0)
            vi_std = np.std(theta_m,axis=0)
            vi_corr,_ = spearmanr(theta_m, axis=0)
            vi_skew = skew(theta_m, axis=0)

            return ELBO_store, vi_mean, vi_std, vi_corr, vi_skew
        
        else:
            return ELBO_store