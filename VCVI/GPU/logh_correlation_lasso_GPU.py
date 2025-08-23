import torch
torch.set_default_dtype(torch.float64)

import math
pi = torch.tensor(math.pi).to('cuda')

from typing import NewType,Tuple
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

from .KVC_utility import GaussianCDF

#%% ----------------------- log posterior ------------------------
class LOGH_CORRELATION:

    # theta = (tau,deltat,gt): parameters to inference
    # nut = tau * exp(deltat) * exp(gt): unconstraint for angles
    # nu <- nut: angles

    def __init__(self, X):
        self.X = X.to('cuda')
        self.XX = X.T @ X
        self.N = X.shape[0]
        self.r = X.shape[1]
        self.dim1 = int( self.r * (self.r-1)/2 )
        self.dim2 = 2 * self.dim1
        self.dim = self.dim2 + 1
    
    def logh(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        thetac = theta.detach().clone().requires_grad_(True)

        tau = thetac[:self.dim1]
        deltat = thetac[self.dim1:self.dim2]
        gt = thetac[self.dim-1]

        # ----------------- log likelihood -----------------------
        # nut = torch.exp(gt) * ( tau * torch.exp(deltat) )  # horseshoe prior
        nut = tau * torch.exp(0.5*deltat)    # Bayes LASSO prior
        nu = 0.03 + (pi - 0.06) * GaussianCDF.apply(nut) # constraint angles

        lower_triangular_indices = torch.tril_indices(self.r, self.r, offset=-1)
        nu_mat = torch.zeros(self.r, self.r).to('cuda') # a matrix to store the angles
        nu_mat[lower_triangular_indices[0], lower_triangular_indices[1]] = nu

        C = torch.cos(nu_mat)
        S = torch.cat([torch.ones(self.r, 1).to('cuda'), torch.sin(nu_mat[:, :-1])], dim=1)
        S_cumprod = torch.cumprod(S, dim=1)

        # Compute Cholesky factor
        L = C * S_cumprod
        I_r = torch.eye(self.r).to('cuda')
        L_inv = torch.linalg.solve_triangular(L, I_r, upper=False)

        # log_likelihood = -self.N * torch.sum( torch.log( torch.diag(L) ) ) - 0.5 * torch.sum( (L_inv.T @ L_inv - I_r) * self.XX )
        # log_likelihood = -self.N * torch.sum( torch.log( torch.diag(L) ) ) - 0.5 * torch.sum( (L_inv.T @ L_inv ) * self.XX )

        xi = L_inv @ self.X.T

        T1 = torch.sum(xi*xi, dim=0)

        logdetL = torch.sum(torch.log(torch.diag(L)))
        # log_likelihood = torch.sum(-logdetL-0.5*(T1-T2))
        log_likelihood = torch.sum(-logdetL - self.r/2 * torch.log(2*pi) - 0.5*(T1))

        # ---------------- log prior -------------------------------
        # bayes lasso prior
        log_prior =  ( torch.sum( -torch.log( torch.sqrt(2*pi) ) - 0.5 * tau**2 ) 
                        + torch.sum( torch.log( torch.exp(gt)/2 ) - torch.exp(gt)/2 * torch.exp(deltat) + deltat )
                        - torch.exp(0.5*gt) + torch.log(torch.tensor(0.5)) + 0.5*gt
                        )

        

        logh_v = log_likelihood + log_prior
        gr_logh = torch.autograd.grad(logh_v, thetac)[0]

        return logh_v.detach(), gr_logh