import jax
import jax.numpy as jnp

#@jax.jit
def rotate_sim(k, sim):
    k = k % 4

    condition1 = (k > 0)
    condition2 = (k > 1)
    condition3 = (k > 2)
    condition4 = (k == 3)

    # if k == 0:
    def kzero(k):
        return sim
    # if k == 1:
    def kone(k):
        return jnp.rot90(sim, k=1, axes=(1,2))
    # if k == 2:
    def ktwo(k):
        return jnp.rot90(sim, k=2, axes=(1,2))
    def kthree(k):
        return jnp.rot90(sim, k=3, axes=(1,2))

    # if >2, use kthree, else use ktwo
    def biggerthantwo(k):
        return jax.lax.cond(condition3, true_fun=kthree, false_fun=ktwo, operand=k)

    # if > 1, return biggerthan2, else use kone
    def biggerthanone(k):
        return jax.lax.cond(condition2, true_fun=biggerthantwo, false_fun=kone, operand=k)

    # if >0 , return biggerthan1, else use kzero
    sim = jax.lax.cond(condition1, true_fun=biggerthanone, false_fun=kzero, operand=k)

    return sim



def run_training(key, noiseamp, weightdir, weightfile, config):

    outdir = config["datadir"]
    NOISEAMP = noiseamp
    print("-----------------------")
    print("RETRAINING FOR NOISEAMP: ", noiseamp)
    print("-----------------------")
    
    # --- IMNN STUFF
    # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip(1.0)]
    transf.append(optax.add_decayed_weights(1e-4))
    optimiser = optax.chain(
        *transf,
        optax.adam(learning_rate=config["learning_rate"])
    )
    
    # ---
    
    # insatiate new moped statistic with higher noise settings
    mymoped, moped_stat, noisevars = get_moped_and_summaries(N=config["N"], 
                                                             noiseamp=NOISEAMP, 
                                                             n_gal=config["ngal"],
                                                             cl_cut=config["cl_cut"], 
                                                             outbins=config["cls_outbins"], 
                                                             bins=None, 
                                                             do_log=False, 
                                                             simdir=config["datadir"],
                                                             L=config["L"]
                                        )

    # initialise model
    strides1 = [1,1]
    strides2 = [1,1]
    strides3 = [1,1]
    strides4 = [1,1]
    strides5 = [1,1]
    
    input_filters1 = [4,1,1]

    
    
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


    num_bins = 6
    cl_shape = indices.shape[0]*num_bins

    np = jnp
    # initialise imnn
    n_s_eff = config["n_s_eff"]
    n_d_eff = config["n_d_eff"] 
    gc.collect()
    IMNN =  newNoiseNumericalGradientIMNN(
        n_s=n_s_eff, n_d=n_d_eff, n_params=config["n_params"], n_summaries=2 + 2,
        input_shape=input_shape, θ_fid=θ_fid, δθ=δθ, model=model,
        optimiser=optimiser, key_or_state=jnp.array(model_key),
        noise_simulator=partial(noise_simulator_dict, 
                                noisescale=NOISEAMP, rot=True),
        chunk_size=2,
        fiducial=outdir + "val_fid.npy", #jnp.load(outdir + "val_fid.npy")[:n_s_eff], #  #
        derivative=outdir + "val_derv.npy",  #jnp.load(outdir + "val_derv.npy")[:n_d_eff], #  
        validation_fiducial=outdir + "fid.npy", #jnp.load(outdir + "fid.npy")[:n_s_eff], # 
        validation_derivative=outdir + "val_derv.npy", #jnp.load(outdir + "derv.npy")[:n_d_eff], #  
        #existing_statistic=None,
        existing_statistic=moped_stat,
        no_invC=True, # True
        evidence=True
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
    IMNN.fit(10.0, 0.01, γ=1.0, rng=jnp.array(rng), print_rate=2, patience=75, max_iterations=7000, min_iterations=200) 

    # print and show fishers
    print("MOPED F: ", mymoped.F)
    print("final IMNN F: ", IMNN.F)
    print("final IMNN_F / MOPED_F :", jnp.linalg.det(IMNN.F) / jnp.linalg.det(mymoped.F)) 

    ax = IMNN.plot(expected_detF=jnp.linalg.det(mymoped.F))
    ax[0].set_yscale("log")
    plt.show()

    # new best weights
    weightfile = "imnn_w_N_%d_2out_noise_%d"%(N, noiseamp*100)

    # save the IMNN weights
    save_obj(IMNN.w, weightdir + weightfile)
    weightfile += ".pkl" # add extension

    # -----
    # make a fisher plot
    mean = θ_fid
    fishers = [mymoped.F, IMNN.F]
    colours =["orange", "black", "blue"]
    labels = [r"Cls, $\ell=%d$"%(6430),  "info-update IMNN with Cls + field"]
    
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
    np.savez("N192_noisy/summaries_noise_%d"%(noiseamp * 100),
             moped_summaries=moped_summaries,
             prior_theta=prior_theta,
             network_outputs=outputs,
             target=target,
             noisy_target=noisy_target,
             target_theta=target_theta,
             network_target=network_target,
             moped_target=moped_target
            )
    gc.collect()

    return weightfile

