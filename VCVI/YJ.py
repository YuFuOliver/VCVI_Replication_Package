# Yu Fu
# A class to define the Yeo-Johnson transformation and its utility functions

import torch
torch.set_default_dtype(torch.float64)

# ---------------- transform to constraint space ----------
def etat2eta(etat):

    eta_max = 2
    eta_min = 0
    eta = (eta_max-eta_min)/(torch.exp(-etat)+1) + eta_min

    return eta

# ------------------ the YJ class ----------------------------------------------------
class YJ:
    """
    YJ element-wise transformation defined on unconstraint parameter space (etat)
    """

    def __init__(self, etat):
        self.etat = etat
        self.eta = etat2eta(etat)

    # ------------------- YJ transformation -------------------
    def iG(self, theta):
        """
        Yeo-Johnson transformation, with limits corrected (theta -> z)
        """
        eta = self.eta

        iG = torch.zeros(theta.shape)

        theta1_index = theta<0
        theta2_index = theta>=0

        theta1 = theta[theta1_index] #theta<0
        theta2 = theta[theta2_index] #theta>=0
        eta1 = eta[theta1_index]
        eta2 = eta[theta2_index]

        def YJ1(theta1,eta1):
            iG1 = - ( torch.pow(1-theta1, 2-eta1) -1 ) / (2-eta1)
            return iG1

        def YJ2(theta2,eta2):
            iG2 = ( torch.pow(theta2+1, eta2) -1 ) / eta2
            return iG2

        iG[theta1_index] = YJ1(theta1,eta1)
        iG[theta2_index] = YJ2(theta2,eta2)

        eta0_index = eta == 0
        eta2_index = eta == 2

        theta1_eta2 = theta1_index & eta2_index
        theta2_eta0 = theta2_index & eta0_index

        iG[theta1_eta2] = - torch.log( 1 - theta[theta1_eta2] )
        iG[theta2_eta0] = torch.log( theta[theta2_eta0] + 1 )

        return iG

    # ------------------- inverse YJ transformation -------------------
    def G(self, z):
        """
        Inverse of Yeo-Johnson transformation, with limits corrected (z -> theta)
        """
        eta = self.eta

        G = torch.zeros(z.shape)

        z1_index = z<0
        z2_index = z>=0

        z1 = z[z1_index] #z<0
        z2 = z[z2_index] #z>=0
        eta1 = eta[z1_index]
        eta2 = eta[z2_index]

        def IYJ1(z1,eta1):
            Gz =  1 - torch.pow( 1 - z1*(2-eta1) , 1/(2-eta1))
            return Gz

        def IYJ2(z2,eta2):
            Gz = torch.pow( 1 + z2*eta2, 1/eta2 ) - 1
            return Gz

        G[z1_index] = IYJ1(z1,eta1)
        G[z2_index] = IYJ2(z2,eta2)

        eta0_index = eta == 0
        eta2_index = eta == 2

        z1_eta2 = z1_index & eta2_index
        z2_eta0 = z2_index & eta0_index

        G[z1_eta2] = 1 - torch.exp(-z[z1_eta2])
        G[z2_eta0] = torch.exp(z[z2_eta0]) - 1

        return G
    
    # Derivative of the YJ transformation w.r.t. theta
    def diG_dtheta(self, theta):
        """ 
        Derivative of the YJ transformation w.r.t. theta
        """
        eta = self.eta
        
        diGdtheta = torch.zeros(theta.shape)

        theta1_index = theta<0
        theta2_index = theta>=0

        theta1 = theta[theta1_index] #theta<0
        theta2 = theta[theta2_index] #theta>=0
        eta1 = eta[theta1_index]
        eta2 = eta[theta2_index]

        def d1(theta1,eta1):
            d = torch.pow(1-theta1, 1-eta1)
            return d

        def d2(theta2,eta2):
            d = torch.pow(theta2+1, eta2-1)
            return d

        diGdtheta[theta1_index] = d1(theta1,eta1)
        diGdtheta[theta2_index] = d2(theta2,eta2)

        return diGdtheta