import math
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import optax
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
import cloudpickle as pickle


np = jnp

from imnns import *
from imnn_update import *
from cls_utils import *
from moped import *

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


#@title script to get moped dependent on noise amplitdue
def get_moped_and_summaries(N, noiseamp, n_gal=30.,
                            ell_min=0, 
                            cl_cut=-1, 
                            outbins=6, 
                            bins=None, 
                            do_log=False, 
                            simdir="/data101/makinen/lemur_sims/pm_sims/N192_hires/",
                            L=250):

    
    cl_cut = cl_cut
    if bins is not None:
        OUTBINS = bins
    else:
        OUTBINS = outbins
    num_tomo = 4
    Lgrid = (L, L, 4000)
    Nmesh = (N,N,512)
    num_bins = outbins
    chi_grid = (jnp.arange(Nmesh[2]) + 0.5) * Lgrid[2] / Nmesh[2]
    chi_source = chi_grid[-1]
    indices = jnp.array(indices_vector(num_tomo))
    cl_shape = indices.shape[0] * num_bins
    
    NOISEAMP = noiseamp
    do_noise = True

    print("COMPUTING MOPED COMPRESSION FOR N=%d, NOISEAMP=%.2f, ELL_MIN=%d"%(N, noiseamp, ell_min))

    # loop function to load all N256 simulations
    fid = jnp.concatenate(jnp.array(
                        [jnp.load(simdir + "fid/sim_%d.npy"%(i))[jnp.newaxis, ...] for i in range(n_s*2)]
                        ), axis=0)
    derv = jnp.concatenate(jnp.array(
                        [jnp.load(simdir + "derv/sim_%d.npy"%(i))[jnp.newaxis, ...] for i in range(n_d*2*2*2)]
                        ), axis=0).reshape(n_d*2, 2, 2, 4, N, N)
    
    key = jr.PRNGKey(7777)
    key,rng = jr.split(key)
    
    fid_keys = jr.split(key, num=2*n_s)
    derv_keys = jr.split(rng, num=2*n_d)
    derv_keys = jnp.repeat(derv_keys, θ_der.shape[0], axis=0) # flattened repeated keys

    # compute the ells and find the argmin
    ell,clsfoo = analysis.compute_auto_cross_angular_power_spectrum(fid[0, 0], fid[0, 0], chi_source, Lgrid[0])
    ellmin = np.argmin((ell - ell_min)**2)

    print("MINIMUM ELL IDX: ", ellmin)

    
    def compute_variance_catalog(zmean=z_means_analysis, n_gal=n_gal):
    
        N0 = Nmesh[0]
        N1 = Nmesh[1]
        N2 = Nmesh[2]
        L0 = Lgrid[0]
        L1 = Lgrid[1]
        L2 = Lgrid[2]
        
        Ncat = 4
    
        cosmo = jc.Planck15(Omega_c=0.3, sigma8=0.8) # no sigma8-dependence 
        rms = 0.3  # from review (corrected w Hall comment)
        a = 1. / (1. + zmean)
        dist = jc.background.radial_comoving_distance(cosmo, a, log10_amin=-3, steps=256)
        angle = 2. * jnp.arctan((L0/N0/2) / dist)
        arcmin_angle = angle * 180. / np.pi * 60.
        arcmin2_pix = arcmin_angle**2
        sources = n_gal / Ncat * arcmin2_pix # from Euclid
        return rms**2 / sources

    noisevars = compute_variance_catalog()

    print("NOISEVARS", noisevars)
    print("SCALED NOISE SIGMAS", jnp.sqrt(noisevars) * noiseamp)

    #@partial(jax.jit, static_argnums=(3,4))
    def noise_simulator_dict(sim, noisescale=noiseamp, rot=True, noisevars=noisevars):
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

    def noise_simulator(key, sim, noisescale=noiseamp, rot=True, noisevars=noisevars):
        key1,key2 = jr.split(key)
        # do rotations of simulations
        k = jr.choice(key1, jnp.array([0,1,2,3]), shape=())
        if rot:
         sim = rotate_sim(k, sim)
        else:
         sim = sim
    
        # now add noise
        # this generates white noise across all pixels and then increases the amplitude
        sim += (jr.normal(key2, shape=(4,N,N)) * noisescale * jnp.sqrt(noisevars).reshape(4,1,1))
        return sim
    

    #### DEFINE FUNCTION TO GET MOPED STATISTIC FOR GIVEN CONFIGURATION
    def get_moped_statistic(noiseamp):
    
        def get_spec(index, tomo_data, key):
            
            if do_noise:
                tomo_data = noise_simulator(key, tomo_data, noisescale=noiseamp)
        
            ell,cl = compute_auto_cross_angular_power_spectrum(tomo_data[index[0]], tomo_data[index[1]],
                                                        chi_source, Lgrid[0])
            return jnp.histogram(ell[:cl_cut], weights=cl[:cl_cut], bins=OUTBINS)[0]
            
        
        def cls_allbins(tomo_data, key, chunk_size=10):
            #gps = lambda i: get_spec(i, tomo_data, key)
            gps = partial(get_spec, tomo_data=tomo_data, key=key)
            return nk.jax.vmap_chunked(gps, chunk_size=chunk_size)(indices)

        
        
        # ----- DO MOPED WITH ALL SIMS (NOISEFREE) -----
        fid_cls = []
        batch = 10
        for i in tq(range(fid.shape[0] // batch)):
            f_ = jax.vmap(cls_allbins)(fid[i*batch:(i+1)*batch], fid_keys[i*batch:(i+1)*batch]).reshape(-1, len(indices)*(num_bins))
            fid_cls.append(f_)
        
        fid_cls = jnp.concatenate(fid_cls)
        
        derv_cls = []
        for i in tq(range(n_d*2*2*n_params // batch)):
            d_  = jax.vmap(cls_allbins)(derv.reshape(-1, num_tomo, Nmesh[0], Nmesh[1])[i*batch:(i+1)*batch], 
                                       derv_keys[i*batch:(i+1)*batch])
            derv_cls.append(d_)
        
        derv_cls = jnp.concatenate(derv_cls).reshape(n_d*2, 2, n_params, len(indices)*(num_bins))
        # ----- -----    

        # DEFINE MOPED OBJECT
        mymoped = MOPED(n_param=2, n_d=n_d*2, n_s=n_s*2, 
                        input_shape=fid_cls[0].shape, 
                        fiducial=fid_cls, 
                        derivatives=derv_cls, 
                        δθ=δθ, 
                        θ_fid=θ_fid
                       )
        
        moped_summs = mymoped.compress(fid_cls)
        
        # next we're going to do a second moped compression to get the derivatives into a smaller space.
        # you can check to see that the fisher is the same for both !
        mymoped2 = MOPED(
                        n_param=2, n_d=n_d*2, n_s=n_s*2, 
                        input_shape=(2,), 
                        fiducial=mymoped.compress(fid_cls), 
                        derivatives=mymoped.compress(derv_cls.reshape((-1,) + fid_cls[0].shape)).reshape(n_d*2, 2, 2, 2), 
                        δθ=δθ, 
                        θ_fid=θ_fid
        )

        moped_statistic = dict(
                n_t = 2,
                mean_derivatives = mymoped2.mu_dervs,
                covariance = mymoped.invF,
                Fisher = mymoped.F,
                fid_summaries=mymoped.compress(fid_cls),
        )
        # spits out dictionary to feed to network
        return mymoped, moped_statistic


    mymoped, moped_statistic = get_moped_statistic(noiseamp)
    return mymoped, moped_statistic, noisevars














