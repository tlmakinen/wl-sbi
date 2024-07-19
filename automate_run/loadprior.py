import numpy as np
from tqdm import tqdm as tq
num = 5000


outdir = "/data101/makinen/lemur_sims/pm_sims/N192_hires/"
# prior_sims = jnp.load(outdir + "prior_S8_L_250_N_128_Nz_512_LPT2.npz")["prior_sims"][:num]
# prior_theta = jnp.load(outdir + "prior_S8_L_250_N_128_Nz_512_LPT2.npz")["prior_theta"][:num]

print("loading new prior sims")
prior_sims = np.concatenate([np.load(outdir + "smaller_prior/sim_%d.npy"%(i))[np.newaxis, ...] for i in tq(range(num))])
prior_theta = np.load("/data101/makinen/lemur_sims/pm_sims/N192_hires/big_prior_theta.npy")

# save the npz file because I forgot to save the parameter draws
np.savez(outdir + "additional_prior_sims", 
          prior_sims = prior_sims,
          prior_theta = prior_theta
         )

print("concatenating all sims into massive file")
# now concatenate to the existing massive prior to load all at once
prior1 = np.load(outdir +"prior_sims.npz" )
prior2 = np.load(outdir + "additional_prior_sims.npz")

prior_theta = np.concatenate([prior1["prior_theta"], prior2["prior_theta"]], axis=0)
prior_sims = np.concatenate([prior1["prior_sims"], prior2["prior_sims"]], axis=0)

print("sims shape", prior_sims.shape)
print("theta shape", prior_theta.shape)


np.savez(outdir + "all_prior_sims", 
          prior_sims = prior_sims,
          prior_theta = prior_theta
         )