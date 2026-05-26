# This file is an example setup to reproduce Cacciato results.
# Follow similar steps to implement a working HOD model.

from .cacciato_inputs import cosmo_pars, cacciato_med_pars, sampleinfo, mag_bins, magnitude_to_luminosity
from .hod import CacciatoHOD
from .plotting import plot_hod_single, plot_hod_grid, plot_phi_L
import pyccl as ccl
import numpy as np

# Setup cosmology and mass def
cosmo = ccl.Cosmology(*cosmo_pars)
logh = np.log10(cosmo['h'])
hmd_200m = ccl.halos.MassDef200m
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)


if __name__ == '__main__':
    # single-bin example
    bright, faint = sampleinfo['mag_bins'][0]
    log_L1 = np.log10(magnitude_to_luminosity(faint)) #log10( L1 / [Lsun/h^2] )
    log_L2 = np.log10(magnitude_to_luminosity(bright)) #log10( L2 / [Lsun/h^2] )
    hod = CacciatoHOD(mass_def=hmd_200m, cM_rel=cM, log_L1=log_L1, log_L2=log_L2, **cacciato_med_pars)
    plot_hod_single(hod, outname='Cacciato_{bright:0.2f}_{faint:0.2f}_HOD_Nc_Ns_vs_M.pdf')

    # grid of bins
    def hod_constructor(log_L1, log_L2):
        params = dict(log_L1=log_L1, log_L2=log_L2, hval=cosmo['h'], **cacciato_med_pars)
        return CacciatoHOD(mass_def=hmd_200m, cM_rel=cM, **params)

    plot_hod_grid(hod_constructor, sampleinfo, magnitude_to_luminosity, outname='Cacciato_all_bins_HOD_Nc_Ns_vs_M.pdf')

    # Phi(L)
    from pyccl.halos.halo_model import HMCalculator
    mass_function = ccl.halos.MassFuncTinker10(mass_def=hmd_200m)
    halo_bias = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
    hmcalc = HMCalculator(mass_function=mass_function, halo_bias=halo_bias, mass_def=hmd_200m)

    log_lum_bin_centers = np.empty(len(sampleinfo['mag_bins']))
    phi_L_cen = np.empty_like(log_lum_bin_centers)
    phi_L_sat = np.empty_like(log_lum_bin_centers)
    for ii, (M2, M1) in enumerate(sampleinfo['mag_bins']):
        log_L2 = np.log10(magnitude_to_luminosity(M2)) #log10( L2 / [Lsun/h^2] )
        log_L1 = np.log10(magnitude_to_luminosity(M1)) #log10( L1 / [Lsun/h^2] )
        log_lum_bin_centers[ii] = 0.5 * (log_L1 + log_L2)
        temp_hod = CacciatoHOD(mass_def=hmd_200m, cM_rel=cM, log_L1=log_L1, log_L2=log_L2, hval=cosmo['h'], **cacciato_med_pars)
        z_mean = sampleinfo['zmeans'][ii]
        phi_L_cen[ii] = hmcalc.integrate_over_massfunc(temp_hod._Nc, cosmo, 1.0/(1.0+z_mean))
        phi_L_sat[ii] = hmcalc.integrate_over_massfunc(temp_hod._Ns, cosmo, 1.0/(1.0+z_mean))

    plot_phi_L(log_lum_bin_centers, phi_L_cen, phi_L_sat, outname='Cacciato_HOD_phi_L_cen_sat.pdf')
    print('Done')
