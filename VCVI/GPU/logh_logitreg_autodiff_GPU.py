import torch
torch.set_default_dtype(torch.float64)

import math
pi = torch.tensor(math.pi)

from scipy.stats import norm

from typing import NewType,Tuple
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

#%% ----------------------- log posterior ------------------------
class LOGH_LOGITREG:

    # theta = (tau,deltat,gt): parameters to inference
    # beta = tau * exp(deltat) * exp(gt) # parameters of logistic regression

    def __init__(self, X, Y):
        self.X = X.to('cuda')
        self.Y = Y.to('cuda')
        self.dim = 2 * X.shape[1] + 1 # dimension of this model
        self.dim1 = int((self.dim - 1) / 2)
        self.dim2 = 2*self.dim1
    
    def logh(self, theta: TensorWithGrad) -> Tuple[torch.Tensor, torch.Tensor]:
        thetac = theta.to('cuda').detach().clone().requires_grad_(True)

        tau = thetac[:self.dim1]
        delta_tilde = thetac[self.dim1:self.dim2]
        xi_tilde = thetac[self.dim-1]

        beta = torch.exp(xi_tilde) * ( tau * torch.exp(delta_tilde) )
        
        X = self.X
        Y = self.Y

        # --------------- log of likelihood ----------------------------------------------------
        def log_logreg(beta,X,Y):
            """
            Compute the log-likelihood and its gradient of logistic regression
            """

            # value
            F = X @ beta
            YF = - Y * F
            mm = torch.where(YF >= 0, YF, 0)
            log_logreg_v = - torch.sum( mm + torch.log(torch.exp(-mm) + torch.exp(YF-mm)) )

            # gradient
            # T = (Y+1)/2
            # S = torch.sigmoid(F)

            # log_logreg_d = X.T @ (T-S)

            return log_logreg_v

        # --------------- log of prior ----------------------------------------------------
        def log_prior(tau,delta_tilde,xi_tilde):
            """
            Compute the log-likelihood  of the prior
            """

            log_prior_v =  torch.log(2/pi) + torch.sum(-torch.log( torch.sqrt(2*pi) ) - 0.5 * tau**2) + torch.sum( torch.log(2/pi) + delta_tilde - torch.log(1 + torch.exp(2*delta_tilde)) ) + xi_tilde - torch.log( 1+ torch.exp(2*xi_tilde))

            return log_prior_v
        
        # --------------- log of h_theta ----------------------------------------------------
        log_logreg_v = log_logreg(beta,X,Y)
        log_prior_v = log_prior(tau,delta_tilde,xi_tilde)

        log_h_v = log_logreg_v + log_prior_v

        # log_h_d1 = log_logreg_d * torch.exp(delta_tilde) * torch.exp(xi_tilde) - tau
        # log_h_d2 = log_logreg_d * beta + 1 - ( ( 2 * torch.exp(2*delta_tilde) ) / ( 1 + torch.exp(2*delta_tilde) ) )
        # log_h_d3 = torch.dot( log_logreg_d, beta ) + 1 - ( ( 2 * torch.exp(2*xi_tilde) ) / ( 1 + torch.exp(2*xi_tilde) ) )
        # log_h_d3 = log_h_d3.unsqueeze(0)
        # log_h_d = torch.cat([log_h_d1,log_h_d2,log_h_d3], dim = 0)

        gr_logh = torch.autograd.grad(log_h_v, thetac)[0]

        # return log_h_v.detach(),log_h_d.detach()
        return log_h_v.detach(), gr_logh