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
import gc


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


import yaml
np = jnp

from functools import partial
import netket as nk
jconfig.update("jax_enable_x64", False) # just to be sure nk doesn't undo the above



from network.moped import *
from fisherplot import plot_fisher_ellipse
from network.net_utils import *
from network.mpk_net192 import *
from network.cls_utils import *
from network.train_utils import rotate_sim
from fisherplot import plot_fisher_ellipse

from lemur import analysis, background, cosmology, limber, simulate, plot, utils, constants




# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


# SHOULD WE UNPACK THE CONFIGS HERE OR IN THE FUNCTION ?


indices = indices_vector(config["num_tomo"])
cl_shape = config["cls_outbins"] * len(indices)

# define the mpk layer before initialising network

dtype = jnp.bfloat16
kernel_size = config["mpk_kernel"]
polynomial_degrees = config["polynomial_degrees"]

mpk_layer = MultipoleCNNFactory(
             kernel_shape=(kernel_size, kernel_size),
             polynomial_degrees=polynomial_degrees,
             output_filters=None,
             dtype=dtype)


act = smooth_leaky


key = jr.PRNGKey(55)
rng, key = jr.split(key)
input_shape = (4, N, N)

# TODO: update config dictionary to take in activation function callable
# config.update({"act": act})


def run_training(key, noiseamp, weightdir, weightfile, config):

    outdir = config["datadir"]
    N = config["N"]
    num_tomo = config["num_tomo"]
    NOISEAMP = noiseamp
    print("-----------------------")
    print("RETRAINING FOR NOISEAMP: ", noiseamp)
    print("-----------------------")
    
    # --- IMNN STUFF
    # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip(config["gradient_clip"])]
    transf.append(optax.add_decayed_weights(1e-4))
    optimiser = optax.chain(
        *transf,
        optax.adam(learning_rate=config["learning_rate"])
    )
    
    # ---
    
    # insatiate new moped statistic with higher noise settings
    key = jr.PRNGKey(7777)
    mymoped, moped_stat, noisevars = get_moped_and_summaries(
                                                             key,
                                                             N=config["N"], 
                                                             noiseamp=NOISEAMP, 
                                                             n_gal=config["ngal"],
                                                             cl_cut=config["cl_cut"], 
                                                             outbins=config["cls_outbins"], 
                                                             bins=None, 
                                                             do_log=False, 
                                                             simdir=config["datadir"],
                                                             L=config["L"]
                                        )

    # initialise mpk model
    model_key = jr.PRNGKey(44)
    model = MPK_InceptNet(
                        filters=config["filters"],
                        # rest of network
                        multipole_tomo1 = MPK_layer(
                                    multipole_layers=[mpk_layer.build_cnn_model(num_input_filters=f,
                                                               strides=config["mpk_strides"],
                                                               pad_size=None) for i,f in enumerate(config["mpk_input_filters"])],
                                    act=config["act"]),
                        moped=mymoped,
                        act=config["act"], 
                        n_outs=2,
                        dtype=jnp.bfloat16
    )

    # def noise_simulator_dict(sim, noisescale=noiseamp, rot=True, noisevars=noisevars):
    #     key = sim["key"] # assigned in IMNN scheme
    #     sim = sim["data"]
    #     key1,key2 = jr.split(key)
    #     # do rotations of simulations
    #     k = jr.choice(key1, jnp.array([0,1,2,3]), shape=())
    #     if rot:
    #      sim = rotate_sim(k, sim)
    #     else:
    #      sim = sim
    #     # now add noise
    #     # this generates white noise across all pixels and then increases the amplitude
    #     sim = sim.at[...].add(jr.normal(key2, shape=(num_tomo,N,N)) * noisescale * jnp.sqrt(noisevars).reshape(num_tomo,1,1))
    #     return sim

    noise_simulator_dict = lambda d: noise_simulator_dict(d, noisescale=noiseamp, rot=True, noisevars=noisevars)
    noise_simulator = lambda k,d: noise_simulator(k, d, noisescale=noiseamp, rot=True, noisevars=noisevars)

    num_bins = config["cls_outbins"]
    cl_shape = indices.shape[0]*num_bins

    np = jnp
    # initialise imnn
    n_s_eff = config["n_s_eff"]
    n_d_eff = config["n_d_eff"] 
    θ_fid = config["θ_fid"]
    δθ = 2*config["δθ"] # 2x for IMNN calculation

    gc.collect()
    IMNN =  newNoiseNumericalGradientIMNN(
        n_s=n_s_eff, n_d=n_d_eff, n_params=config["n_params"], 
        n_summaries=config["n_params"] + config["n_extra_summaries"], # output is 2 moped + 2 new summaries
        input_shape=input_shape, θ_fid=θ_fid, δθ=δθ, model=model,
        optimiser=optimiser, key_or_state=jnp.array(model_key),
        noise_simulator=partial(noise_simulator_dict, 
                                noisescale=NOISEAMP, rot=True),
        chunk_size=2,
        fiducial=outdir + "val_fid.npy",
        derivative=outdir + "val_derv.npy",
        validation_fiducial=outdir + "fid.npy",
        validation_derivative=outdir + "val_derv.npy",
        existing_statistic=moped_stat,
        no_invC=True,
        evidence=True,
        do_reg=False
    )
    gc.collect()
    # load the previous round's weights
    wbest = load_obj(weightdir + weightfile)
    print("num trainable params: ", sum(x.size for x in jax.tree_util.tree_leaves(wbest)))
    
    IMNN.set_F_statistics(wbest, key)
    print("MOPED F: ", mymoped.F)
    print("initial IMNN F: ", IMNN.F)
    print("initial det IMNN F: ", np.linalg.det(IMNN.F))
    print("initial IMNN_F / MOPED_F :", jnp.linalg.det(IMNN.F) / jnp.linalg.det(mymoped.F)) 

    print("training IMNN now")
    key,rng = jax.random.split(key) # retrain # patience=75, min_its=300 
    IMNN.fit(10.0, 0.01, γ=1.0, rng=jnp.array(rng), 
                                print_rate=2, 
                                patience=config["patience"], 
                                max_iterations=config["max_iterations"], 
                                min_iterations=config["min_iterations"]) 

    # print and show fishers
    print("MOPED F: ", mymoped.F)
    print("final IMNN F: ", IMNN.F)
    print("final IMNN_F / MOPED_F :", jnp.linalg.det(IMNN.F) / jnp.linalg.det(mymoped.F)) 

    ax = IMNN.plot(expected_detF=jnp.linalg.det(mymoped.F))
    ax[0].set_yscale("log")
    plt.show()

    # new best weights
    weightfile = config["weightfile"] +  "_N_%d_noise_%d"%(N, noiseamp*100)

    # save the IMNN weights
    save_obj(IMNN.w, weightdir + weightfile)
    weightfile += ".pkl" # add extension

    # -----
    # make a fisher plot
    mean = config["θ_fid"]
    fishers = [mymoped.F, IMNN.F]
    colours =["orange", "black", "blue"]
    labels = ["Cls",  "info-update IMNN with Cls + field"]
    
    for i,f in enumerate(fishers): 
        if i==0:
            ax = plot_fisher_ellipse(f, mean=mean, color=colours[i], label=labels[i])
        else:
            plot_fisher_ellipse(f, mean=mean, ax=ax, color=colours[i], label=labels[i])
    
    plt.legend(framealpha=0.0)
    plt.xlabel(r'$\Omega_m$')
    plt.ylabel(r'$S_8$')
    plt.show()
    # -----

    
    print("COMPRESSING PRIOR")
    # pull in all the prior simulations
    prior_sims = jnp.load("/data101/makinen/lemur_sims/pm_sims/N192_hires/smaller_prior_sims.npz")["prior_sims"]
    prior_theta = jnp.load("/data101/makinen/lemur_sims/pm_sims/N192_hires/smaller_prior_sims.npz")["prior_theta"]
    
    def get_sigma8(omegam, S8):
        return S8 / (jnp.sqrt(omegam / 0.3))
    
    def get_S8(theta):
        return np.array([theta[:, 0], theta[:, 1]*np.sqrt(theta[:, 0]/0.3)]).T

    
    key = jr.PRNGKey(333)
    noisekeys = jr.split(key, num=prior_sims.shape[0])
    # add in noise
    def _assign_keys(key, data):
        return dict(key=key,
                    data=data)
        
    prior_sims = jax.vmap(_assign_keys)(noisekeys, prior_sims)
    prior_sims = nk.jax.vmap_chunked(noise_simulator_dict, chunk_size=2)(prior_sims)
    
    # now compute Cls 
    prior_cls = []
    batch = 10
    for i in tq(range(prior_sims.shape[0] // batch)):
        f_ = jax.vmap(cls_allbins_nonoise)(prior_sims[i*batch:(i+1)*batch]).reshape(-1, len(indices)*(num_bins))
        prior_cls.append(f_)
    
    prior_cls = jnp.concatenate(prior_cls)
    # compress with moped to get Cls summaries
    moped_summaries = mymoped.compress(prior_cls)

    # get IMNN to compress
    batch = 500
    outputs = jnp.concatenate([IMNN.get_estimate(prior_sims[i*batch:(i+1)*batch]) for i in range(prior_sims.shape[0] // batch)])

    # load target
    target = jnp.load(outdir + "target_L_%d_N_%d_Nz_512.npz"%(L, N))["kappa"]
    target_theta = jnp.load(outdir + "target_L_%d_N_%d_Nz_512.npz"%(L, N))["theta"]
    
    noise_target_key = jax.random.PRNGKey(604)
    noisy_target = noise_simulator(noise_target_key, target, rot=True, noisevars=noisevars, noisescale=noiseamp)
    network_target = IMNN.get_estimate(noisy_target[jnp.newaxis, ...])
    moped_target = mymoped.compress(cls_allbins_nonoise(noisy_target).reshape(-1, len(indices)*(num_bins)))

    # plot summaries
    plt.subplot(121)
    im = plt.scatter(prior_theta[:, 0], outputs[:, 0], s=2, c=prior_theta[:, 1])
    plt.axvline(θ_fid[0], c="k", ls="--")
    plt.scatter(target_theta[0], network_target[:, 0], c="orange", marker="*", s=74, zorder=23)
    
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"MOPED + net summary 1")
    plt.colorbar(im)
    
    plt.subplot(122)
    im = plt.scatter(prior_theta[:, 1], outputs[:, 1], s=2, c=prior_theta[:, 0])
    plt.axvline(θ_fid[1], c="k", ls="--")
    plt.scatter(target_theta[1], network_target[:, 1], c="orange", marker="*", s=74, zorder=23)
    plt.ylabel(r"MOPED + net summary 2")
    plt.xlabel(r"$\sigma_8$")
    plt.colorbar(im)
    plt.tight_layout()

    plt.show()
    
    print("saving everything")
    outfile_name = config["output_directory"] + "summaries_noise_%d"%(noiseamp * 100)
    np.savez(outfile_name,
             moped_summaries=moped_summaries,
             prior_theta=prior_theta,
             network_outputs=outputs,
             target=target,
             noisy_target=noisy_target,
             target_theta=target_theta,
             network_target=network_target,
             moped_target=moped_target,
             moped_F=mymoped.F,
             network_F=IMNN.F
            )
    gc.collect()

    return weightfile










