# Yu Fu 28 Oct 2024
# Kendall vector copula utility functions

import torch
torch.set_default_dtype(torch.float64)
from torch.autograd import Function
from scipy.stats import norm,erlang

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