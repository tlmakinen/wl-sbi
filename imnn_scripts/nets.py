from typing import Any, Callable, Sequence, Optional, Union
from flax.core import freeze, unfreeze
import flax.linen as nn

import jax
import jax.numpy as jnp

Array = Any


class InceptBlock(nn.Module):
    """Inception block submodule"""
    filters: Sequence[int]
    strides: Union[None, int, Sequence[int]]
    dims: int
    do_5x5: bool = True
    do_3x3: bool = True
    #input_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):

        outs = []
        
        if self.do_5x5:
        # 5x5 filter
          x1 = nn.Conv(features=self.filters[0], kernel_size=(1,)*self.dims, strides=None)(x)
          x1 = nn.Conv(features=self.filters[0], kernel_size=(5,)*self.dims, strides=self.strides)(x1)
          outs.append(x1)
          
        if self.do_3x3:
        # 3x3 filter
          x2 = nn.Conv(features=self.filters[1], kernel_size=(1,)*self.dims, strides=None)(x)
          x2 = nn.Conv(features=self.filters[1], kernel_size=(3,)*self.dims, strides=self.strides)(x2)
          outs.append(x2)

        # 1x1
        x3 = nn.Conv(features=self.filters[2], kernel_size=(1,)*self.dims, strides=None)(x)
        x3 = nn.Conv(features=self.filters[2], kernel_size=(1,)*self.dims, strides=self.strides)(x3)
        outs.append(x3)
        
        # maxpool and avgpool
        x4 = nn.max_pool(x, (3,)*self.dims, padding='SAME')
        x4 = nn.Conv(features=self.filters[3], kernel_size=(1,)*self.dims, strides=self.strides)(x4)
        outs.append(x4)
                    
        x = jnp.concatenate(outs, axis=-1)
        
        return x   


class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.swish(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


class InceptNet(nn.Module):
    """An incept net architecture"""
    filters: Sequence[int]
    mlp_features: Sequence[int]
    div_factor: float = 0.02
    n_params: int = 2
    act: Callable = nn.swish
    
    @nn.compact
    def __call__(self, x):
        dim = 2

        # add in Cls information
        # cls = cls_allbins(jax.lax.stop_gradient(x)).reshape(-1)
        # cls = jax.lax.stop_gradient(jnp.log(cls))
        # cls += 20.0
        # cls = MLP(self.mlp_features)(cls)

        # do field-level net
        x = jnp.transpose(x, (1, 2, 0))
        x /= self.div_factor # so that we can use learning_rate=1e-3
        x += 1.5
        x = x[..., jnp.newaxis] # expand dims for 3D conv   

        fs = 4
        fs2 = 4
        
        x = InceptBlock([fs, fs2, fs2, fs], strides=(4,4,2), dims=3, do_5x5=False)(x) # 32
        x = nn.swish(x)
        

        x = InceptBlock([fs, fs2*2, fs2*2, fs], strides=(4,4,2), dims=3, do_5x5=False)(x) # 16
        x = self.act(x)
        x = jnp.squeeze(x) # SQUEEZE for DIM=2
        x = InceptBlock([fs, fs2*8, fs2*8, fs], strides=2, dims=2, do_5x5=False)(x) # 4
        x = self.act(x)
        x = InceptBlock([fs, fs2*16, fs2*16, fs], strides=2, dims=2, do_5x5=False)(x) # 2
        x = self.act(x)
        x = InceptBlock([fs, fs2*16, fs2*16, fs], strides=2, dims=2, do_5x5=False, do_3x3=False)(x) # 1
        x = self.act(x)     
        x = nn.Conv(features=self.n_params, kernel_size=(1,)*dim, strides=None)(x)
        x = x.reshape(-1)
        
        # combine Cls information
        # x = jnp.concatenate([x, cls])
        # x = nn.Dense(50)(x)
        # x1 = nn.swish(x)
        # x = nn.Dense(50)(x1)
        # x = nn.swish(x + x1)
        # x = nn.Dense(self.n_params)(x)
        
        return x
