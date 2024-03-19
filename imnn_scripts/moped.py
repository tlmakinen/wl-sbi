# we know that the moped summaries are really informative, so let's add them in as an inductive bias
import numpy as onp
import jax
import jax.numpy as jnp


class MOPED:
    
    def __init__(self,
                 n_param,
                 n_d,
                 n_s,
                 input_shape,
                 fiducial, 
                 derivatives,
                 δθ,
                 θ_fid,
                ):
        """MOPED compression scheme as described in https://arxiv.org/abs/astro-ph/9911102
        and generalised in https://arxiv.org/abs/1712.00012.

        Parameters
        ----------
        n_s : int
        Number of simulations used to calculate summary covariance
        n_d : int
        Number of simulations used to calculate mean of summary
        derivative with respect to the model parameters
        n_params : int
        Number of model parameters to compress to
        input_shape : int
        Number of summaries, e.g. dimensionality of covariance
        fiducial : float(n_s, input_shape)
        Summaries computed at fiducial point for covariance estimation
        derivatives: float(n_d, 2, n_params, input_shape)
        Summaries computed at finite difference points for derivative estimation
        δθ : float(n_params,)
        Step size between derivative finite differences
        θ_fid : float(n_params,)
        Fiducial point at which the covariance is calculated


        Methods
        -------
        compress:
        Function to compress input data using the MOPED Fisher-weighted attributes
        _get_mu_derivatives:
        Function which reshapes derivatives and calculates their means from finite
        differences
        _get_fisher:
        Function which calculates the Fisher information of the MOPED summaries
        """
        
        self.n_d = n_d
        self.n_s = n_s
        self.δθ = δθ
        self.θ_fid = θ_fid
        self.n_param = n_param
        
        self.C = jnp.cov(fiducial.reshape(-1, fiducial.shape[-1]), rowvar=False)
        self.invC = jnp.linalg.inv(self.C)
        self.mu = jnp.mean(fiducial, 0)
        self.mu_dervs = self._get_mu_derivatives(derivatives)
        
        self.F = self._get_fisher()
        self.invF = jnp.linalg.inv(self.F)
        
        
    def _get_mu_derivatives(self, derivatives):
        x_mp = derivatives.reshape(self.n_d, 2, self.n_param, derivatives.shape[-1])
        _dervs = (x_mp[:, 1, :, :] - x_mp[:, 0, :, :]) / jnp.expand_dims(jnp.expand_dims(self.δθ, 0), -1)
        
        return jnp.mean(_dervs, 0)
    
    def compress(self, x):
        delta = x - self.mu
        
        summaries = jnp.einsum("mi,ij,jk->mk", 
                               self.mu_dervs, 
                               self.invC, 
                               delta.T).T
        
        return self.θ_fid + jnp.einsum(
                    "ij,mj->mi",
                    self.invF,
                    summaries
                                )
    
    
    def _get_fisher(self):
        return 0.5 * jnp.einsum("ij,jl,kl->ik", self.mu_dervs, self.invC, self.mu_dervs)
    