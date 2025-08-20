# Yu Fu
# reparameterization trick for KVC_spline and KVC_SVUC

import numpy as np
from scipy.stats import norm,erlang,expon
import torch
torch.set_default_dtype(torch.float64)
from torch.autograd import Function

from typing import NewType
TensorWithGrad = NewType('TensorWithGrad', torch.Tensor)

from .YJ import YJ

#%% ----------------------- Gaussian CDF ------------------------
class GaussianCDF(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.tensor(norm.cdf(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * torch.tensor(norm.pdf(input))
        return grad_input

# Example usage
# x = torch.tensor([0.335, 0.5, 0.9], requires_grad=True)
# cdf = GaussianCDF.apply(x)
# cdf.sum().backward()
# print(x.grad)
# print('check',norm.pdf(x.detach()))

#%% ---------------------- Gaussian ICDF ------------------------
class GaussianICDF(Function):
    @staticmethod
    def forward(ctx, input):
        ICDF = torch.tensor(norm.ppf(input))
        ctx.save_for_backward(input, ICDF)
        return ICDF

    @staticmethod
    def backward(ctx, grad_output):
        input, ICDF = ctx.saved_tensors
        pdf = torch.tensor(norm.pdf(ICDF))
        grad_input = grad_output / pdf
        return grad_input

# Example usage
# u = torch.tensor([0.1, 0.6, 0.9], requires_grad=True)
# x = GaussianICDF.apply(u)
# x.sum().backward()
# print(u.grad)
# print('check',1/norm.pdf(x.detach()))

#%% ---------------------- Erlang ICDF ------------------------
class ErlangICDF(Function):
    @staticmethod
    def forward(ctx, input, shape):
        ICDF = torch.tensor(erlang.ppf(input, shape))

        ctx.save_for_backward(input,ICDF)
        ctx.shape = shape
        return ICDF

    @staticmethod
    def backward(ctx, grad_output):
        input,ICDF = ctx.saved_tensors
        shape = ctx.shape

        pdf = torch.tensor(erlang.pdf(ICDF, shape))
        grad_input = grad_output / pdf
        return grad_input, None
    
# Example usage
# V = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
# d = torch.tensor([3,12,4])  # Shape parameter for Erlang distribution
# R = ErlangICDF.apply(1-V,d)
# R.sum().backward()
# print(V.grad)

# R = R.detach()
# fR = erlang.pdf(R,d)
# dRdV =  - (1 / fR)
# print('check',dRdV)


#%% -------------------- reparameterization trick --------------------
def reptrick(tau: TensorWithGrad, b:TensorWithGrad, nu:TensorWithGrad,
             L: TensorWithGrad, yj: YJ, d: np.ndarray) -> tuple[TensorWithGrad,TensorWithGrad,torch.Tensor]:
    
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
    theta = b + (nu**2) * yj.G(L @ x)

    if C.grad_fn is not None:
        return theta,C,dlogdetC_dtau
    else:
        return theta

# example usage
# tau = torch.eye(3, requires_grad=True)
# d = torch.tensor([1,1,1])  # Shape parameter for Erlang distribution
# b = torch.zeros(3, requires_grad = True)
# l = torch.ones(3, requires_grad = True)
# theta = reptrick(tau,b,l,d)

# theta.sum().backward()
# print(tau.grad)
# print(b.grad)
# print(l.grad)