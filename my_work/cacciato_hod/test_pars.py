# one used by Christos, with b_1 and b_2 rescaled by h=0.739
# I'm keeping this only to compare my HOD plots from his.
christos_pars = dict(
    log_L1=np.log10(1.5e7), # not a correct bin value
    log_L2=np.log10(5.6e9), # not a correct bin value
    log_L0=9.95,
    log_M1=11.24,
    gamma_1=3.18,
    gamma_2=0.245,
    sigma_c=0.157,
    alpha_s=-1.18,
    b_0=-1.17,
    b_1=1.53 * 0.739, # this is wrong
    b_2=-0.217 * 0.739**2, # this is wrong
)
