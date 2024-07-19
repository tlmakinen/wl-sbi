import os,sys,pathlib
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
import numpy as onp

from functools import partial
import flax.linen as nn
import jax.random as jr
import cloudpickle as pickle
import gc
from tqdm import tqdm as tq
import yaml
from functools import partial
import netket as nk
jconfig.update("jax_enable_x64", False) # just to be sure nk doesn't undo the above
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local imports here
from network.moped import *
from fisherplot import plot_fisher_ellipse
from network.net_utils import *
from network.mpk_net192 import *
from network.cls_utils import *
from network.train_utils import *
from network.imnn_update import *
from fisherplot import plot_fisher_ellipse
from lemur import analysis, background, cosmology, limber, simulate, plot, utils, constants

np = jnp


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# -----
# folder to load config file
CONFIG_PATH = "./config/"

config_name = sys.argv[1]

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config(config_name)
# -----


# -----
# UNPACK GLOBAL CONFIG VALUES
outdir = config["datadir"]
N = config["N"]
L = config["L"]
num_tomo = config["num_tomo"]    

indices = jnp.array(indices_vector(config["num_tomo"]))
cl_shape = config["cls_outbins"] * len(indices)


num_bins = config["cls_outbins"]
chi_grid = (jnp.arange(config["Nz"]) + 0.5) * config["Lz"] / float(config["Nz"])
cl_cut = config["cl_cut"]
chi_source = chi_grid[-1]

# cls function to pass to network
def cls_allbins_nonoise(tomo_data, chunk_size=2):
    def get_spec_nonoise(index, tomo_data):
        ell,cl = compute_auto_cross_angular_power_spectrum(tomo_data[index[0]], tomo_data[index[1]],
                                                chi_source, config["L"])
        return jnp.histogram(ell[:cl_cut], weights=cl[:cl_cut], bins=num_bins)[0]
    gps = partial(get_spec_nonoise, tomo_data=tomo_data)
    
    return nk.jax.vmap_chunked(gps, chunk_size=chunk_size)(indices)

# define the mpk layer before initialising network

dtype = jnp.bfloat16
kernel_size = config["mpk_kernel"]
polynomial_degrees = config["polynomial_degrees"]
do_moped = config["do_moped"]

mpk_layer = MultipoleCNNFactory(
            kernel_shape=(kernel_size, kernel_size),
            polynomial_degrees=polynomial_degrees,
            output_filters=None,
            dtype=dtype)


act = smooth_leaky

key = jr.PRNGKey(55)
rng, key = jr.split(key)
input_shape = (num_tomo, N, N)
# -----

# whether or not we're doing moped compression
if do_moped:
    n_t_summaries = config["n_params"]
else:
    n_t_summaries = cl_shape


def run_compression(key, noiseamp, 
                 weightdir, 
                 weightfile, 
                 config):

    # make output directories
    pathlib.Path(config["output_directory"]).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(config["output_plot_directory"]).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(config["weightdir"]).mkdir(parents=True, exist_ok=True) 


    # dump the config file with noise extension (in case we change training params)
    outconfig = os.path.join(config["weightdir"], 'noise_%d'%(noiseamp*100) + config_name)
    with open(outconfig, 'w') as file:
        yaml.dump(config, file)


    NOISEAMP = noiseamp
    print("-----------------------")
    print("COMPRESSING FOR NOISEAMP: ", noiseamp)
    print("-----------------------")

    if not do_moped:
        print("-----------------------")
        print("USING CLS AS SUMMARIES, NOT MOPED ")
        print("-----------------------")


    if weightfile is not None:
        print("WILL LOAD WEIGHTS FROM FILE", weightdir + weightfile)
        print("-----------------------")


    # --- OPTIMISER STUFF
    # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip(float(config["gradient_clip"]))]
    transf.append(optax.add_decayed_weights(1e-4))
    optimiser = optax.chain(
        *transf,
        optax.adam(learning_rate=float(config["learning_rate"]))
    )
    # ---
    
    # insatiate new moped statistic with higher noise settings
    key = jr.PRNGKey(7777)
    θ_fid = jnp.array(config["θ_fid"])
    δθ = 2*jnp.array(config["δθ"]) # 2x for IMNN calculation
    num_bins = jnp.array(config["cls_outbins"])
    cl_shape = indices.shape[0]*num_bins
    mymoped, moped_stat, noisevars = get_moped_and_summaries(
                                                             key,
                                                             N=config["N"], 
                                                             noiseamp=NOISEAMP, 
                                                             n_s=config["n_s"],
                                                             n_d=config["n_d"],
                                                             n_params=config["n_params"],
                                                             θ_fid=θ_fid,
                                                             δθ=δθ,
                                                             z_means=jnp.array(config["z_means"]),
                                                             n_gal=config["ngal"],
                                                             cl_cut=config["cl_cut"], 
                                                             outbins=config["cls_outbins"], 
                                                             bins=None, 
                                                             do_log=False, 
                                                             simdir=config["datadir"],
                                                             L=config["L"],
                                                             do_moped=do_moped
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
                                    act=act),
                        moped=mymoped,
                        cl_compression=cls_allbins_nonoise,
                        act=act, 
                        cl_shape=cl_shape,
                        n_outs=2,
                        dtype=jnp.bfloat16,
                        do_moped=do_moped
    )

    # wrap the noise simulators with appropriate noise amplitude
    # noise simulator for chunking with nk.vmap_chunked (pass key to dict object in imnn routine)
    def noise_simulator_dict(sim, noisescale=noiseamp, noisevars=noisevars, rot=True):
        key = sim["key"] # assigned in IMNN scheme
        sim = sim["data"]
        key1,key2 = jr.split(key)
        # do rotations of simulations
        k = jr.choice(key1, jnp.array([0,1,2,3]), shape=())
        if rot:
            sim = rotate_sim(k, sim)
        else:
            sim = sim
        # now add noise
        # this generates white noise across all pixels and then increases the amplitude
        sim = sim.at[...].add(jr.normal(key2, shape=(4,N,N)) * noisescale * jnp.sqrt(noisevars).reshape(4,1,1))
        return sim


    # default noise simulator
    noise_sim = lambda k,d: noise_simulator(k, d, noisescale=noiseamp, rot=True, noisevars=noisevars, N=N, num_tomo=num_tomo)


    np = jnp
    # initialise imnn
    n_s_eff = config["n_s_eff"]
    n_d_eff = config["n_d_eff"] 

    gc.collect()
    IMNN =  newNoiseNumericalGradientIMNN(
        n_s=n_s_eff, n_d=n_d_eff, 
        n_params=config["n_params"], 
        n_summaries=n_t_summaries + config["n_extra_summaries"], # output is 2 moped + 2 new summaries
        input_shape=input_shape, θ_fid=θ_fid, δθ=δθ, model=model,
        optimiser=optimiser, 
        key_or_state=jnp.array(model_key),
        noise_simulator=noise_simulator_dict,
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
    print("LOADING WEIGHTS FROM FILE", weightdir + weightfile)
    wbest = load_obj(weightdir + weightfile)


    IMNN.set_F_statistics(wbest, key)
    print("num trainable params: ", sum(x.size for x in jax.tree_util.tree_leaves(wbest)))
    # print and show fishers
    print("-----------------------")
    print("MOPED F: ", mymoped.F)
    print("IMNN F: ", IMNN.F)
    print("det IMNN F: ", np.linalg.det(IMNN.F))
    print("IMNN_F / MOPED_F :", jnp.linalg.det(IMNN.F) / jnp.linalg.det(mymoped.F)) 
    print("-----------------------")


    def compress_prior(name="prior_big", keyint=333):
    
        print("COMPRESSING PRIOR: ", name)
        # pull in all the prior simulations
        prior_sims = jnp.load(config[name])["prior_sims"]
        prior_theta = jnp.load(config[name])["prior_theta"]
        
        def get_sigma8(omegam, S8):
            return S8 / (jnp.sqrt(omegam / 0.3))
        
        def get_S8(theta):
            return np.array([theta[:, 0], theta[:, 1]*np.sqrt(theta[:, 0]/0.3)]).T

        
        key = jr.PRNGKey(keyint)
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

        # get IMNN to compress prior
        batch = 500
        outputs = jnp.concatenate([IMNN.get_estimate(prior_sims[i*batch:(i+1)*batch]) for i in range(prior_sims.shape[0] // batch)])


        return prior_theta, moped_summaries, outputs
    
    # compress both priors
    prior_theta_big, moped_summs_big, network_summs_big = compress_prior("prior_big")
    prior_theta_small, moped_summs_small, network_summs_small = compress_prior("prior_small")


    # load target
    target = jnp.load(outdir + "target_L_%d_N_%d_Nz_512.npz"%(L, N))["kappa"]
    target_theta = jnp.load(outdir + "target_L_%d_N_%d_Nz_512.npz"%(L, N))["theta"]
    
    noise_target_key = jax.random.PRNGKey(604)
    noisy_target = noise_sim(noise_target_key, target)
    network_target = IMNN.get_estimate(noisy_target[jnp.newaxis, ...])
    moped_target = mymoped.compress(cls_allbins_nonoise(noisy_target).reshape(-1, len(indices)*(num_bins)))
    
    
    # --- plot summaries over large prior ---
    plt.subplot(121)
    im = plt.scatter(prior_theta_big[:, 0], network_summs_big[:, 0], s=2, c=prior_theta_big[:, 1])
    plt.axvline(θ_fid[0], c="k", ls="--")
    plt.scatter(target_theta[0], network_target[:, 0], c="orange", marker="*", s=74, zorder=23)
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"MOPED + net summary 1")
    plt.colorbar(im)
    
    plt.subplot(122)
    im = plt.scatter(prior_theta_big[:, 1], network_summs_big[:, 1], s=2, c=prior_theta_big[:, 0])
    plt.axvline(θ_fid[1], c="k", ls="--")
    plt.scatter(target_theta[1], network_target[:, 1], c="orange", marker="*", s=74, zorder=23)
    plt.ylabel(r"MOPED + net summary 2")
    plt.xlabel(r"$\sigma_8$")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(config["output_plot_directory"] + "summary_scatter_bigprior_noiseamp_%d"%(noiseamp*100))
    plt.close()
    # --- ---


    print("saving everything")

    outfile_name = os.path.join(config["output_directory"], "summaries_noise_%d_compression"%(noiseamp * 100))
    np.savez(outfile_name,
            # small prior
             moped_summaries_small=moped_summs_small,
             prior_theta_small=prior_theta_small,
             network_outputs_small=network_summs_small,

            # big prior
             moped_summaries_big=moped_summs_big,
             prior_theta_big=prior_theta_big,
             network_outputs_big=network_summs_big,

            # target simulation
             target=target,
             noisy_target=noisy_target,
             target_theta=target_theta,
             network_target=network_target,
             moped_target=moped_target,

             # IMNN and MOPED fishers
             moped_F=mymoped.F,
             network_F=IMNN.F
            )
    gc.collect()

    return weightfile





def main():

    key = jr.PRNGKey(444)
    
    noiseamp = 1.0
    noiseamp_to_load = 1.0
    weightfile = config["weightfile"] +  "_N_%d_noise_%d.pkl"%(N, noiseamp_to_load*100)

    weightfile = run_compression(key, noiseamp, 
                    weightdir=config["weightdir"], 
                    weightfile=weightfile, 
                    config=config)
    
    

if __name__ == "__main__":
    main()
