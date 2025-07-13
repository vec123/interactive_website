import gpytorch
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class GraphKernel(gpytorch.kernels.Kernel):
    def __init__(self, laplacian, **kwargs):
        super().__init__(**kwargs)
        self.laplacian = laplacian

    def forward(self, x1, x2, **params):
        return torch.exp(-torch.mm(x1 @ self.laplacian, x2.T))
    
    
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_points = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_points, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        # Initialise X with PCA or randn
        if pca == True:
             X_init = _init_pca(Y, latent_dim) # Initialise X to PCA
        else:
             X_init = torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        
        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)





class MultiTaskbGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_points = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, self.inducing_points, q_u, learn_inducing_locations=True
            ),
            num_tasks=data_dim,
            num_latents=data_dim,
            latent_dim=-1,
        )
        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        # Initialise X with PCA or randn
        if pca == True:
             X_init = _init_pca(Y, latent_dim) # Initialise X to PCA
        else:
             X_init = torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
        



class badly_def_model_0(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, laplacian):
        super().__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.MultitaskMean(
         #   gpytorch.means.ConstantMean(), num_tasks=train_y.size(-1)
        #)
        #----------mean module 
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(
                prior=gpytorch.priors.NormalPrior(loc=0.0, scale=1.0)
            ),
            num_tasks=train_y.size(-1)
        )
        
        #----------covar module 
         # Temporal multi-scale kernels
        short_term_kernel = gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 1.0))
        short_term_kernel.lengthscale = 0.1  # Short-term scale
        medium_term_kernel = gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 1.0))
        medium_term_kernel.lengthscale = 1.0  # Medium-term scale
        long_term_kernel = gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 1.0))
        long_term_kernel.lengthscale = 5.0  # Long-term scale

        temporal_kernel = short_term_kernel + medium_term_kernel + long_term_kernel + gpytorch.kernels.PeriodicKernel()
        spatial_kernel = gpytorch.kernels.ScaleKernel( gpytorch.kernels.RBFKernel() + GraphKernel(laplacian) )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            spatial_kernel,
            num_tasks=train_y.shape[1],
            rank=13,
        )

    def forward(self,  joints):
        mean_x = self.mean_module(joints)
        covar_x = self.covar_module(joints)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class badly_def_model_1(gpytorch.models.ApproximateGP):
    def __init__(self, num_data, latent_dim, likelihood, laplacian):
        # Variational distribution and strategy
        inducing_points = torch.randn(num_data, latent_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()

        spatial_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        graph_kernel = GraphKernel(laplacian)
        self.covar_module = gpytorch.kernels.ProductKernel(spatial_kernel, graph_kernel)

  

    def forward(self, z):
        mean_z = self.mean_module(z)
        covar_z = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_z, covar_z)

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.enc_fc_mu = nn.Linear(64*7*7, latent_dim)
        self.enc_fc_logvar = nn.Linear(64*7*7, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 64*7*7)
        self.dec_deconv1 = nn.ConvTranspose2d(64,32,4,2,1)
        self.dec_deconv2 = nn.ConvTranspose2d(32,1,4,2,1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0),-1)
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(-1,64,7,7)
        x = F.relu(self.dec_deconv1(x))
        x = torch.sigmoid(self.dec_deconv2(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar
