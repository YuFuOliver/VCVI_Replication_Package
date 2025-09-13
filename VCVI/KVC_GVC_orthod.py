# Yu Fu
# Gaussian vector copula with an orthogonal correlation matrix + KVC
# Note that this version can take both StanModel and custom log posterior function as input
# margins are identity matrices
# this version is specifically designed for shrinkage priors such that dim = 2m+1. (horseshoe prior)
# Change the YJ functionality to calculate computational time accurately (10 Oct 2024)
# use median of stable variational parameters to be consistent with reported ELBO, use spearman correlation

from scipy.stats import norm,skew,expon,spearmanr
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import bridgestan as bs
import math
pi = torch.tensor(math.pi)

from typing import NewType, Tuple, Callable, Optional
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

from .YJ import YJ
from .KVC_utility import GaussianCDF, GaussianICDF, ErlangICDF

# ------------------------- l(unconstraint) to Lam (-1 to 1) --------------------------------
def l2Lam(l):
    Lam_max = 1
    Lam_min = -1
    Lam = (Lam_max-Lam_min)/(torch.exp(-l)+1) + Lam_min

    return Lam



# %%
# ----------------------- KVC+GVC class --------------------------------

class KVC_GVC_orthod:

    @staticmethod
    def rep_trick(tau: TensorWithGrad, b: TensorWithGrad, l: TensorWithGrad, 
              s: TensorWithGrad, etat: TensorWithGrad,
              dim1:torch.Tensor, d: np.ndarray, isYJ) -> tuple[TensorWithGrad,TensorWithGrad,torch.Tensor]:
        
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
        x1 = x[:dim1]
        x2 = x[dim1:2*dim1]
        x3 = x[2*dim1]

        Lam = l2Lam(l)
        
        z1 = x1
        z2 = Lam * x1 + torch.sqrt(1 - Lam**2) * x2
        z3 = x3
        z3 = torch.unsqueeze(z3, dim=0)
        
        z = torch.cat([z1,z2,z3], dim = 0)

        if isYJ:
            theta = b + s**2 * ( YJ(etat).G( z ) )
        else:
            theta = b + s**2 * z

        if C.grad_fn is not None:
            return theta,C,dlogdetC_dtau
        else:
            return theta


    def __init__(self, d, is_vc, isYJ, optimizer, sampling, stan_model: bs.StanModel, log_post: Optional[Callable] = None):

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

        self.num_group = d.size
        self.d = d

        # Initialize variational parameters
        self.tau = torch.eye(self.num_group, requires_grad=True)
        self.b = torch.zeros(self.dim, requires_grad=True)
        self.l = torch.zeros(self.dim1, requires_grad=True)
        self.s = torch.full((self.dim,), 0.5, requires_grad=True)
        self.etat = torch.zeros(self.dim, requires_grad = True) # unconstraint eta

        vari_para = [self.b, self.l, self.s]
        
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

        # Reparameterization trick
        theta,C,dlogdetC_dtau = KVC_GVC_orthod.rep_trick(self.tau, self.b, self.l, self.s, self.etat, self.dim1, self.d, self.isYJ)

        return theta,C,dlogdetC_dtau
    
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

    def logq(self, C: TensorWithGrad, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        # For the chain rule: dELBO/dtheta * dtheta/dlambda:
        # clone theta to thetac so that variational parameters only contribute to theta
        # detach variational parameters to avoid gradient tracking in logq computation
        
        # ----------------- deal with the variational parameters --------------------------
        dim = self.dim

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

        logq_v = log_p + log_det - 0.5*torch.logdet(C@C.T)

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
        tau_store = torch.zeros(1000, self.num_group, self.num_group)
        b_store = torch.zeros(1000, self.dim)
        l_store = torch.zeros(1000, self.dim1)
        s_store = torch.zeros(1000, self.dim)
        etat_store = torch.zeros(1000, self.dim)

        for iter in range(num_iter):

            if iter >= num_iter - 1000:
                with torch.no_grad():
                    tau_store[iter - (num_iter - 1000),:,:] = ( torch.diag( torch.abs( torch.diag(self.tau.clone().detach())) )
                                                                + torch.tril(self.tau.clone().detach(),diagonal=-1) )
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
            avg_tau = torch.median(tau_store, dim=0)
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
                theta = KVC_GVC_orthod.rep_trick(avg_tau, avg_b, avg_l, avg_s, avg_etat, self.dim1, self.d, self.isYJ)
                theta_m[i,:] = theta.numpy()
            
            vi_mean = np.mean(theta_m,axis=0)
            vi_std = np.std(theta_m,axis=0)
            vi_corr,_ = spearmanr(theta_m, axis=0)
            vi_skew = skew(theta_m, axis=0)

            return ELBO_store, vi_mean, vi_std, vi_corr, vi_skew
        
        else:
            return ELBO_store