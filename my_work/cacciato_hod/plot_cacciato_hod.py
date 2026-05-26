# This file is an example setup to reproduce Cacciato results.
# Follow similar steps to implement a working HOD model.

from cacciato_inputs import cosmo_pars, cacciato_med_pars, sampleinfo, mag_bins, magnitude_to_luminosity
from hod import CacciatoHOD
from plotting import plot_hod_single, plot_hod_grid, plot_phi_L_vs_L, plot_phi_L_vs_M
import pyccl as ccl
import numpy as np

# Setup cosmology and mass def
cosmo = ccl.Cosmology(**cosmo_pars)
logh = np.log10(cosmo_pars['h'])
hmd_200m = ccl.halos.MassDef200m
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

def hod_constructor(log_L1, log_L2, hval=None, clfpars=None):
    if hval is None:
        hval = cosmo_pars['h']
        print(f"Using default h-val={hval:0.4f}")
    if clfpars is None:
        print(f"Using median Cacciato+2013 HOD pars...")
        clfpars = cacciato_med_pars
    params = dict(log_L1=log_L1, log_L2=log_L2, hval=hval, **clfpars)
    return CacciatoHOD(mass_def=hmd_200m, cM_rel=cM, **params)

if __name__ == '__main__':
    outdir = "plots"

    # single-bin example
    bright, faint = sampleinfo['mag_bins'][0]
    log_L1 = np.log10(magnitude_to_luminosity(faint)) # Lsun/h^2
    log_L2 = np.log10(magnitude_to_luminosity(bright)) # Lsun/h^2
    hod = hod_constructor(log_L1=log_L1, log_L2=log_L2)
    plot_hod_single(hod, outname=f'{outdir}/Cacciato_{bright:0.2f}_{faint:0.2f}_HOD_Nc_Ns_vs_M.pdf', save_png=True)

    # HOD plots for all samples
    plot_hod_grid(hod_constructor, sampleinfo, magnitude_to_luminosity, outname=f'{outdir}/Cacciato_all_bins_HOD_Nc_Ns_vs_M.pdf', save_png=True)
