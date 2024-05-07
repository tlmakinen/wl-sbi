from typing import *

from jax import random, numpy as jnp
from flax import linen as nn
from .multipole_cnn import MultipoleConv
from .multipole_cnn_factory import MultipoleCNNFactory


# class NPE(nn.Module):
#     num_distributions: int
#     weight_dim: int
#     multipole_cnn: MultipoleConv
#     activation_function: Callable

#     def setup(self):
#         self.mdn = MDN(self.num_distributions, self.weight_dim)

#     def __call__(self, input_field):
#         log_density = jnp.log(input_field + 1)
#         psi = self.multipole_cnn(log_density).reshape(self.multipole_cnn.num_output_filters, input_field.size)
#         return self.mdn(self.activation_function(psi))


def build_npe(kernel_shape, polynomial_degrees, num_distributions, activation_function):
    factory = MultipoleCNNFactory(kernel_shape=kernel_shape,
                                  polynomial_degrees=polynomial_degrees,
                                  num_input_filters=1,
                                  output_filters=None)

    cnn_model = factory.build_cnn_model()

    npe = cnn_model

    dummy_input = jnp.arange(125).reshape([5, 5, 5])
    key = random.PRNGKey(42)
    init_model_params = npe.init(key, dummy_input)

    return npe, init_model_params

