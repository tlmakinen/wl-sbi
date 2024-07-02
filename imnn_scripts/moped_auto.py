import math
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import optax
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn

import jax.random as jr

import cloudpickle as pickle


np = jnp

from imnns import *
from imnn_update import *

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


z_means_analysis = jnp.array([0.5, 0.75, 1.0, 1.25]) #jnp.array([0.5, 1.0, 1.5, 2.0])

def indices_vector(num_tomo):
   indices = []
   cc = 0
   for catA in range(0,num_tomo,1):
      for catB in range(catA,num_tomo,1):
        indices.append([catA, catB])
        cc += 1
   return indices

N = 128
L = 250 #250
cl_cut = -1 #14 #13 #6 # 13
skip = 12 #2

OUTBINS = 6

num_tomo = 4
Lgrid = (L, L, 4000)
Nmesh = (N,N,512)
num_bins = jnp.ones(Nmesh[0]//2)[:cl_cut:skip].shape[0]
chi_grid = (jnp.arange(Nmesh[2]) + 0.5) * Lgrid[2] / Nmesh[2]
chi_source = chi_grid[-1]
indices = jnp.array(indices_vector(num_tomo))
cl_shape = indices.shape[0] * cl_cut

# NO NOISE FOR NOW
NOISEAMP = 0.25 #1.0 #0.25


do_noise = True

def compute_variance_catalog(zmean=z_means_analysis):

    N0 = Nmesh[0]
    N1 = Nmesh[1]
    N2 = Nmesh[2]
    L0 = Lgrid[0]
    L1 = Lgrid[1]
    L2 = Lgrid[2]
    
    Ncat = 4

    cosmo = jc.Planck15(Omega_c=0.3, sigma8=0.8) # no sigma8-dependence 
    rms = 0.3 / 2. # from review (corrected w Hall comment)
    a = 1. / (1. + zmean)
    dist = jc.background.radial_comoving_distance(cosmo, a, log10_amin=-3, steps=256)
    angle = 2. * jnp.arctan((L0/N0/2) / dist)
    arcmin_angle = angle * 180. / np.pi * 60.
    arcmin2_pix = arcmin_angle**2
    sources = 30. / Ncat * arcmin2_pix # from Euclid
    return rms**2 / sources

noisevars = compute_variance_catalog()






def get_moped_and_summaries(N, noiseamp, n_gal=30.,
                            ell_min=0, cl_cut=-1, outbins=6, bins=None, do_log=False, L=250):

    
    cl_cut = cl_cut
    skip = 12
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

    # LOAD IN SIMULATIONS AND FIND ELL_MIN 
    fid = jnp.load(outdir + "fid_S8_L_%d_N_%d_Nz_512_pm_Om_%d_s8_%d.npy"%(L, N, θ_fid[0]*10, θ_fid[1]*10)) 
    derv = jnp.load(outdir + "derv_S8_smallstep_L_%d_N_%d_Nz_512_pm_Om_%d_s8_%d.npy"%(L, N, θ_fid[0]*10, θ_fid[1]*10))[:n_d*2]
    
    key = jr.PRNGKey(7777)
    key,rng = jr.split(key)
    
    fid_keys = jr.split(key, num=2*n_s)
    derv_keys = jr.split(rng, num=2*n_d)
    derv_keys = jnp.repeat(derv_keys, θ_der.shape[0], axis=0) # flattened repeated keys

    # compute the ells and find the argmin
    ell,clsfoo = analysis.compute_auto_cross_angular_power_spectrum(fid[59, 0], fid[59, 0], chi_source, Lgrid[0])
    ellmin = np.argmin((ell - ell_min)**2)

    # optional log-spaced bins
    #if bins is not None:
    #    OUTBINS = jnp.logspace(start=jnp.log10(ell[ellmin]), stop=jnp.log10(ell[cl_cut]), num=7)


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
        rms = 0.3 #/ 2. # from review (corrected w Hall comment)
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


    def get_spec_nonoise(index, tomo_data):
        ell,cl = analysis.compute_auto_cross_angular_power_spectrum(tomo_data[index[0]], tomo_data[index[1]],
                                                    chi_source, Lgrid[0])
        if do_log:
            ell = jnp.log(ell)
        return jnp.histogram(ell[ellmin:cl_cut], weights=cl[ellmin:cl_cut], bins=OUTBINS)[0]
    
        
    def cls_allbins_nonoise(tomo_data):
        gps = lambda i: get_spec_nonoise(i, tomo_data)
        return jax.vmap(gps)(indices)

    #### DEFINE FUNCTION TO GET MOPED STATISTIC FOR GIVEN CONFIGURATION
    def get_moped_statistic(noiseamp):
    
        def get_spec(index, tomo_data, key):
            
            tomo_data = noise_simulator(key, tomo_data, noisescale=noiseamp)
            ell,cl = analysis.compute_auto_cross_angular_power_spectrum(tomo_data[index[0]], tomo_data[index[1]],
                                                        chi_source, Lgrid[0])
            if do_log:
                ell = jnp.log(ell)
        
            return jnp.histogram(ell[ellmin:cl_cut], weights=cl[ellmin:cl_cut], bins=OUTBINS)[0]
        
        def cls_allbins(tomo_data, key):
            gps = lambda i: get_spec(i, tomo_data, key)
            return jax.vmap(gps)(indices)

        
        
        # ----- DO MOPED WITH ALL SIMS (NOISEFREE) -----
        fid_cls = []
        batch = 50
        for i in tq(range(fid.shape[0] // batch)):
            f_ = jax.vmap(cls_allbins)(fid[i*batch:(i+1)*batch], fid_keys[i*batch:(i+1)*batch]).reshape(-1, len(indices)*(num_bins))
            fid_cls.append(f_)
        
        fid_cls = jnp.concatenate(fid_cls)
        
        derv_cls = []
        batch = 50
        for i in tq(range(n_d*2*2*n_params // batch)):
            d_  = jax.vmap(cls_allbins)(derv.reshape(-1, num_tomo, Nmesh[0], Nmesh[1])[i*batch:(i+1)*batch], 
                                       derv_keys[i*batch:(i+1)*batch])
            derv_cls.append(d_)
        
        derv_cls = jnp.concatenate(derv_cls).reshape(n_d*2, 2, n_params, len(indices)*(num_bins))
        # ----- -----
        
        cl_shape = derv_cls.shape[-1]
    
    
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
        
        print("moped 2 F:", mymoped2.F, "moped 1 F:", mymoped.F)
        print("moped 2 F:", jnp.linalg.det(mymoped2.F), "moped 1 F:", jnp.linalg.det(mymoped.F))
        
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

    return mymoped, moped_statistic


    

























