import os
from typing import Tuple


# change JAX GPU memory preallocation fraction
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


from jax import config as jconfig
jconfig.update("jax_enable_x64", False)
import jax

import math
import jax.numpy as jnp
import jax_cosmo as jc
import optax
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
import jax.random as jr
import cloudpickle as pickle

import yaml
np = jnp

from functools import partial
from lemur import analysis, background, cosmology, limber, simulate, plot, utils, constants
from moped import *

import netket as nk
jconfig.update("jax_enable_x64", False) # just to be sure nk doesn't undo the above



# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config




# define the mpk layer before initialising network

dtype = jnp.bfloat16

mpk_layer = MultipoleCNNFactory(
             kernel_shape=(13,13),
            #kernel_shape=(7,7),
             polynomial_degrees=[0,1,2],
             output_filters=None,
             dtype=dtype)


strides1 = [1,1]
strides2 = [1,1]
strides3 = [1,1]
strides4 = [1,1]
strides5 = [1,1]

input_filters1 = [4,1,1]
input_filters_rest = [4, 6]
act = smooth_leaky


key = jr.PRNGKey(55)
rng, key = jr.split(key)
input_shape = (4, N, N)

# TODO: update config dictionary to take in activation function callable
# config.update({"act": act})













