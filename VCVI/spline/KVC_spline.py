# Yu Fu
# Kendall vector copula
# Note that this version can take both StanModel and custom log posterior (as a class) as input
# margins constructed for the spline model
# Change the YJ functionality to calculate computational time accurately (28 Oct 2024)
# use median of stable variational parameters to be consistent with reported ELBO, use spearman correlation

import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import bridgestan as bs
from scipy.stats import norm,skew,expon,spearmanr
import math
pi = torch.tensor(math.pi)

from typing import NewType, Tuple, Callable, Optional
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

from VCVI.YJ import YJ
from VCVI.KVC_utility import GaussianCDF, GaussianICDF, ErlangICDF

class KVC_spline:

    @staticmethod
    def reptrick(tau: TensorWithGrad, b:TensorWithGrad, nu:TensorWithGrad,
             L: TensorWithGrad, yj: YJ, d: np.ndarray, isYJ:bool) -> tuple[TensorWithGrad,TensorWithGrad,torch.Tensor]:
    
        """
        Reparameterization trick to generate theta.

        Except from d and yj, all of the tensors are embedded with autodiff.

        Note that yj is an instance of YJ class.
        """

        num_group = tau.size()[0]

        # ------------- eps -> V -----------------------------
        eps = norm.rvs(size=num_group)
        eps = torch.tensor(eps)

        tauL = torch.diag( (torch.diag(tau))**2 ) + torch.tril(tau,diagonal=-1)
        scale = 1/(torch.sqrt( torch.diag(tauL@tauL.T) ))
        C = torch.diag(scale) @ tauL

        # compute dlogdetC_dtau and then clear the effect of this backward
        if C.grad_fn is not None:
            logdetC = torch.logdet(C)
            logdetC.backward(retain_graph=True)
            dlogdetC_dtau = tau.grad.clone()
            tau.grad.zero_()

        # continue the computational graph
        z = C @ eps
        V = GaussianCDF.apply(z)

        # ------------- V -> R -----------------------------
        R = ErlangICDF.apply(1-V,d)

        # ------------- R -> U -----------------------------
        U = torch.empty(0)

        for i in range(num_group):
            Ri = R[i]

            Si_temp = expon.rvs(size=d[i])
            Si = Si_temp/np.sum(Si_temp)
            Si = torch.as_tensor(Si)

            Ui = torch.exp(-Ri * Si)
            U = torch.cat((U,Ui))

        # ------------- U -> theta -------------------------
        x = GaussianICDF.apply(U)

        # theta = b + D @ yj.G(L @ x)
        # theta = b + torch.exp(nu) * yj.G(L @ x)

        if isYJ:
            theta = b + nu**2 * ( yj.G( L @ x ) )
        else:
            theta = b + nu**2 * ( L @ x )

        if C.grad_fn is not None:
            return theta,C,dlogdetC_dtau
        else:
            return theta
        

    def __init__(self, d, is_vc, isYJ, optimizer, sampling, 
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
            self.dim1 = log_post.dim1
            self.dim2 = log_post.dim2
            self.dim = log_post.dim
        else:
            self.dim = self.model.param_num()  # Set dimensionality from StanModel
        
        self.num_group = d.size
        self.d = d

        self.dim1 = d[0]+d[1]+d[2] # the dim of betas
        self.dim2 = self.dim - self.dim1 # the dim of the rest

        # Initialize variational parameters
        self.tau = torch.eye(self.num_group, requires_grad=True)
        self.b = torch.zeros(self.dim, requires_grad=True)
        self.nu = torch.full((self.dim,), 0.5, requires_grad=True)
        self.l12 = torch.zeros(d[0]-1, requires_grad = True)
        self.l22 = torch.zeros(d[1]-1, requires_grad = True)
        self.l32 = torch.zeros(d[2]-1, requires_grad = True)
        self.l13 = torch.zeros(d[0]-2, requires_grad = True)
        self.l23 = torch.zeros(d[1]-2, requires_grad = True)
        self.l33 = torch.zeros(d[2]-2, requires_grad = True)
        self.etat = torch.zeros(self.dim, requires_grad = True)

        vari_para = [self.b, self.nu, self.l12, self.l22, self.l32, self.l13, self.l23, self.l33]

        if is_vc:
            vari_para.append(self.tau)
        
        if isYJ:
            vari_para.append(self.etat)
        
        # Define the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(vari_para, lr=0.001, maximize=True)
        else:
            self.optimizer = optim.Adadelta(vari_para, rho=0.95, maximize=True)
                                            

    def sample(self) -> Tuple[TensorWithGrad,TensorWithGrad,torch.Tensor]:
        # deal with the variational parameters

        L1_inv = torch.diag(torch.ones(self.d[0])) + torch.diag(self.l12, -1) + torch.diag(self.l13, -2)
        L1 = torch.linalg.solve_triangular(L1_inv, self.I1 , upper = False, unitriangular = True)

        L2_inv = torch.diag(torch.ones(self.d[1])) + torch.diag(self.l22, -1) + torch.diag(self.l23, -2)
        L2 = torch.linalg.solve_triangular(L2_inv, self.I2 , upper = False, unitriangular = True)

        L3_inv = torch.diag(torch.ones(self.d[2])) + torch.diag(self.l32, -1) + torch.diag(self.l33, -2)
        L3 = torch.linalg.solve_triangular(L3_inv, self.I3 , upper = False, unitriangular = True)

        L4 = torch.diag( torch.ones(self.dim2) )

        L = torch.block_diag(L1, L2, L3, L4)

        yj = YJ(self.etat)

        # Reparameterization trick
        theta,C,dlogdetC_dtau = KVC_spline.reptrick(self.tau, self.b, self.nu, L, yj, self.d, self.isYJ)

        return theta,C,dlogdetC_dtau
    
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

    def logq(self, C: TensorWithGrad, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        # For the chain rule: dELBO/dtheta * dtheta/dlambda:
        # clone theta to thetac so that variational parameters only contribute to theta
        # detach variational parameters to avoid gradient tracking in logq computation
        
        # ----------------- deal with the variational parameters --------------------------

        with torch.no_grad():            

            L1_inv = torch.diag(torch.ones(self.d[0])) + torch.diag(self.l12, -1) + torch.diag(self.l13, -2)
            L1_inv_ = L1_inv.detach()

            L2_inv = torch.diag( torch.ones(self.d[1]) ) + torch.diag(self.l22, -1) + torch.diag(self.l23, -2)
            L2_inv_ = L2_inv.detach()

            L3_inv = torch.diag( torch.ones(self.d[2]) ) + torch.diag(self.l32, -1) + torch.diag(self.l33, -2)
            L3_inv_ = L3_inv.detach()

            L4_inv = torch.diag( torch.ones(self.dim2) )
            L_inv_ = torch.block_diag(L1_inv_, L2_inv_, L3_inv_, L4_inv) # Note that |L_inv| = 1 = |L|

            b_ = self.b.detach()
            nu_ = self.nu.detach()
            etat_ = self.etat.detach()
        
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
        
        logq_v = logJ - 0.5*torch.logdet(C@C.T)

        gr_logq = torch.autograd.grad(logq_v, thetac)[0]

        return logq_v.detach(), gr_logq

    
    def train_step(self) -> torch.Tensor:
        theta,C,dlogdetC_dtau = self.sample()

        logh_v, gr_logh = self.logh(theta)
        logq_v, gr_logq = self.logq(C,theta)

        with torch.no_grad():
            ELBO = logh_v - logq_v
        
        gr_ELBO = gr_logh - gr_logq
        
        theta.backward(gr_ELBO)
        self.tau.grad = self.tau.grad + dlogdetC_dtau

        self.optimizer.step()
        self.optimizer.zero_grad()

        return ELBO


    def train(self, num_iter=10000):

        np.random.seed(33)

        ELBO_store= torch.zeros(num_iter)

        tau_record = torch.zeros((1000,self.num_group,self.num_group))
        b_record = torch.zeros((1000,self.dim))
        nu_record = torch.zeros((1000,self.dim))
        l12_record = torch.zeros((1000,self.d[0]-1))
        l22_record = torch.zeros((1000,self.d[1]-1))
        l32_record = torch.zeros((1000,self.d[2]-1))
        l13_record = torch.zeros((1000,self.d[0]-2))
        l23_record = torch.zeros((1000,self.d[1]-2))
        l33_record = torch.zeros((1000,self.d[2]-2))
        etat_record = torch.zeros((1000,self.dim))

        for iter in range(num_iter):

            if iter >= num_iter - 1000:
                with torch.no_grad():
                    tau_record[iter - (num_iter - 1000),:,:] = ( torch.diag( torch.abs(torch.diag(self.tau.clone().detach())) )
                                                                + torch.tril(self.tau.clone().detach(),diagonal=-1) )
                    b_record[iter - (num_iter - 1000)] = self.b.detach().clone()
                    nu_record[iter - (num_iter - 1000)] = self.nu.detach().clone()
                    l12_record[iter - (num_iter - 1000)] = self.l12.detach().clone()
                    l22_record[iter - (num_iter - 1000)] = self.l22.detach().clone()
                    l32_record[iter - (num_iter - 1000)] = self.l32.detach().clone()
                    l13_record[iter - (num_iter - 1000)] = self.l13.detach().clone()
                    l23_record[iter - (num_iter - 1000)] = self.l23.detach().clone()
                    l33_record[iter - (num_iter - 1000)] = self.l33.detach().clone()
                    etat_record[iter - (num_iter - 1000)] = self.etat.detach().clone()
            
            # return ELBO (last step) and update variational parameters
            ELBO = self.train_step() # key step!!!

            with torch.no_grad():
                ELBO_store[iter] = ELBO

            # if iter % 100 == 0:
            #     print(f"Iteration {iter}: ELBO = {ELBO.item()}")

        with torch.no_grad():
            avg_tau,_ = torch.median(tau_record, dim=0)
            avg_b,_ = torch.median(b_record, dim=0)
            avg_nu,_ = torch.median(torch.abs(nu_record), dim=0)
            avg_l12,_ = torch.median(l12_record, dim=0)
            avg_l22,_ = torch.median(l22_record, dim=0)
            avg_l32,_ = torch.median(l32_record, dim=0)
            avg_l13,_ = torch.median(l13_record, dim=0)
            avg_l23,_ = torch.median(l23_record, dim=0)
            avg_l33,_ = torch.median(l33_record, dim=0)
            avg_etat,_ = torch.median(etat_record, dim=0)

            L1_inv = torch.diag(torch.ones(self.d[0])) + torch.diag(avg_l12, -1) + torch.diag(avg_l13, -2)
            L1 = torch.linalg.solve_triangular(L1_inv, self.I1 , upper = False, unitriangular = True)

            L2_inv = torch.diag( torch.ones(self.d[1]) ) + torch.diag(avg_l22, -1) + torch.diag(avg_l23, -2)
            L2 = torch.linalg.solve_triangular(L2_inv, self.I2 , upper = False, unitriangular = True)

            L3_inv = torch.diag( torch.ones(self.d[2]) ) + torch.diag(avg_l32, -1) + torch.diag(avg_l33, -2)
            L3 = torch.linalg.solve_triangular(L3_inv, self.I3 , upper = False, unitriangular = True)

            L4 = torch.diag( torch.ones(self.dim2) )
            L = torch.block_diag(L1, L2, L3, L4)

            yj = YJ(avg_etat)

        if self.sampling:

            # sample from variational density
            print('Sampling from variational density...')
            sample_size = 100000
            theta_m = np.zeros((sample_size, self.dim))

            for i in range(sample_size):
                theta = KVC_spline.reptrick(avg_tau,avg_b,avg_nu,L,yj,self.d,self.isYJ)
                theta_m[i,:] = theta.numpy()
            
            vi_mean = np.mean(theta_m,axis=0)
            vi_std = np.std(theta_m,axis=0)
            vi_corr,_ = spearmanr(theta_m, axis=0)
            vi_skew = skew(theta_m, axis=0)

            return ELBO_store, vi_mean, vi_std, vi_corr, vi_skew
        
        else:
            return ELBO_store