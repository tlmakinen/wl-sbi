from typing import Any, Callable, Sequence, Optional, Union
from flax.core import freeze, unfreeze
import flax.linen as nn

import jax
import jax.numpy as jnp
from network.NPE import npe
from network.NPE.multipole_cnn import MultipoleConv
from network.NPE.multipole_cnn_factory import MultipoleCNNFactory
import cloudpickle as pickle

Array = Any
np = jnp

from network.cls_utils import *
from network.moped_auto import *
from network.moped import *
from network.net_utils import *


class InceptStride(nn.Module):
    """Inception block submodule"""
    filters: Sequence[int]
    pad_shape: int
    act: Callable = smooth_leaky
    do_1x1: bool = True
    do_4x4: bool = True
    dim: int = 2
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):

        d = self.dim
        # 2x2 stride
        x2 = nn.Conv(features=self.filters[0], kernel_size=(2,)*self.dim, 
                     strides=(2,)*d if d == 2 else (2,2,1), 
                     padding="CIRCULAR", dtype=self.dtype)(x)
        z = self.act(x2)
        x2 = nn.Conv(features=self.filters[0], kernel_size=(2,)*d, strides=(1,)*d, 
                     padding="CIRCULAR", dtype=self.dtype)(z)
        x2 = self.act(x2 + z)

        # 3x3 stride
        x3 = nn.Conv(features=self.filters[1], kernel_size=(2,)*d, 
                     strides=(3,)*d if d == 2 else (3,3,1), 
                     padding="CIRCULAR", dtype=self.dtype)(x)
        z = self.act(x3)
        x3 = nn.Conv(features=self.filters[1], kernel_size=(2,)*d, strides=(1,)*d, 
                     padding="CIRCULAR", dtype=self.dtype)(z)
        x3 = self.act(x3 + z)

        # 4x4 stride
        if self.do_4x4:
            x4 = nn.Conv(features=self.filters[1], kernel_size=(2,)*d, 
                         strides=(4,)*d if d == 2 else (4,4,1), 
                         padding="CIRCULAR")(x)
            z = self.act(x4)
            x4 = nn.Conv(features=self.filters[1], kernel_size=(2,)*d, strides=(1,)*d, 
                         padding="CIRCULAR", dtype=self.dtype)(z)
            x4 = self.act(x4 + z)

        
        # now pad the 3x3 and concatenate it to the 2x2 stride
        x3shape = conv_outs(W=self.pad_shape)
        pads = get_padding(x3shape, x2.shape[0])
        
        x3 = jnp.pad(x3, 
                     pad_width=(pads, pads, (0,0),) if d == 2 else (pads, pads, (0,0), (0,0)), 
                     mode="wrap")
        x2 = jnp.concatenate([x2, x3], -1)

        # run another conv for the concated versions
        x2 = nn.Conv(features=self.filters[0], kernel_size=(2,)*d, strides=(1,)*d, 
                     padding="CIRCULAR", dtype=self.dtype)(x2)
        x2 = self.act(x2)
        

        # optional no stride embedding
        if self.do_1x1:
            x1 = nn.Conv(features=self.filters[2], kernel_size=(2,)*d, strides=(1,)*d, 
                         padding="CIRCULAR", dtype=self.dtype)(x)
            z = self.act(x1)
            x1 = nn.Conv(features=self.filters[2], kernel_size=(2,)*d, strides=(1,)*d, 
                         padding="CIRCULAR", dtype=self.dtype)(z)
            x1 = self.act(x1 + z)

            return x1, x2, x4

        elif self.do_4x4:
            return x2, x4

        else:
            return x2



class MPK_InceptNet(nn.Module):
    """An incept-stride net architecture with MultiPole Kernel (MPK) embedding"""
    filters: Sequence[int]
    multipole_tomo1: MPK_layer
    moped: MOPED
    cl_compression: Callable
    div_factor: float = 0.02
    cl_shape: int = 60
    n_outs: int = 1
    act: Callable = smooth_leaky
    dtype: Any = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x):

        filters = self.filters

        # add in Cls information
        cls_summs = self.cl_compression(jax.lax.stop_gradient(x.astype(self.dtype))).reshape(-1, self.cl_shape)
        cls_summs = self.moped.compress(jax.lax.stop_gradient(cls_summs)).reshape(-1) # moped compression

        xlog = (log_transform(jax.lax.stop_gradient(x)) / 0.02).transpose((1,2,0))
        x = xlog + 1.0

        x = x.astype(self.dtype)
        # embed the data in multipoles
        x = self.multipole_tomo1(x)
        
        x_1_64, x_1_32 = InceptStride(filters=[1,1,1], pad_shape=192, act=self.act, do_1x1=False, dtype=self.dtype)(x)
        
        x_2_32, x_2_16 = InceptStride(filters=filters, pad_shape=96, act=self.act, do_1x1=False, dtype=self.dtype)(x_1_64)
        # now concatenate the remaining 32x32 arrays
        x_2_32 = jnp.concatenate([x_1_32, x_2_32], -1)
        # now cut down to 16x16
        filters = [f*2 for f in filters]
        x_3_16, x_2_8 = InceptStride(filters=filters, pad_shape=48, act=self.act, do_1x1=False, dtype=self.dtype)(x_2_32)
        # concatenate the remaining 16x16 array
        x_3_16 = jnp.concatenate([x_2_16, x_3_16], -1)
        # now cut down to 8x8
        filters = [f*2 for f in filters]
        x_3_8 = InceptStride(filters=filters, pad_shape=24, act=self.act, do_1x1=False, do_4x4=False, dtype=self.dtype)(x_3_16)
        # OUTPUT IS 12x12
        x = jnp.concatenate([x_2_8, x_3_8], -1)
        # mean pool out
        x = x.mean(axis=(0,1))
        x = x.reshape(-1)
        x = nn.Dense(self.n_outs, dtype=self.dtype)(x).reshape(1,-1)
        x = x.reshape(-1).astype(jnp.float32) # make sure output is float32

        return jnp.concatenate([cls_summs.reshape(-1), x])



