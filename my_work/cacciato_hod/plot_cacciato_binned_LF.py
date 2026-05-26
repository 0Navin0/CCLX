# This file is an example setup to reproduce Cacciato results.
# Follow similar steps to implement a working HOD model.

from plotting import plot_phi_L_vs_L, plot_phi_L_vs_M, plot_binned_LF
import pyccl as ccl
import numpy as np

from plot_cacciato_hod import (
    cacciato_med_pars, sampleinfo,
    mag_bins, magnitude_to_luminosity, cosmo, logh, hmd_200m,
    cM, hod_constructor,
)

from pyccl.halos.halo_model import HMCalculator
mass_function = ccl.halos.MassFuncTinker10(mass_def=hmd_200m)
halo_bias = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
hmcalc = HMCalculator(mass_function=mass_function, halo_bias=halo_bias, mass_def=hmd_200m)

if __name__ == '__main__':

    outdir = "plots"

    # Phi(L)

    log_lum_bin_centers = np.empty(len(sampleinfo['mag_bins']))
    mag_bin_centers = np.empty(len(sampleinfo['mag_bins']))
    phi_L_cen = np.empty_like(log_lum_bin_centers)
    phi_L_sat = np.empty_like(log_lum_bin_centers)
    phi_L_cen_per_mag = np.empty_like(log_lum_bin_centers)
    phi_L_sat_per_mag = np.empty_like(log_lum_bin_centers)

    # integral over HMF requires HOD to handle masses in Msun
    for ii, (M2, M1) in enumerate(sampleinfo['mag_bins']):
        log_L2 = np.log10(magnitude_to_luminosity(M2)) #log10( L2 / [Lsun/h^2] )
        log_L1 = np.log10(magnitude_to_luminosity(M1)) #log10( L1 / [Lsun/h^2] )
        log_lum_bin_centers[ii] = 0.5 * (log_L1 + log_L2)
        mag_bin_centers[ii] = 0.5 * (M1 + M2)
        temp_hod = hod_constructor(log_L1=log_L1, log_L2=log_L2, clfpars=cacciato_med_pars)
        z_mean = sampleinfo['zmeans'][ii]
        # below we compute the number density of cen/sat gal in a
        # lum bin. It's a pure number divided by volume
        phi_L_cen[ii] = hmcalc.integrate_over_massfunc(temp_hod._Nc, cosmo, 1.0/(1.0+z_mean)) # Mpc^-3
        phi_L_sat[ii] = hmcalc.integrate_over_massfunc(temp_hod._Ns, cosmo, 1.0/(1.0+z_mean)) # Mpc^-3

    fsat = phi_L_sat / (phi_L_cen + phi_L_sat)
    # passing log_lum_bin_centers in log10( L / [Lsun/h^2] ) 
    # and phi_L_cen/sat in (Phi / [h^3/Mpc^3])
    phi_L_cen = phi_L_cen /temp_hod.hval**3 # h^3/Mpc^3
    phi_L_sat = phi_L_sat /temp_hod.hval**3 # h^3/Mpc^3
    plot_phi_L_vs_L(log_lum_bin_centers, phi_L_cen, phi_L_sat, fsat, outname=f'{outdir}/Cacciato_HOD_phi_L_cen_sat_vs_L.pdf')

    # plot in Cacciato units
    delta_mag = np.diff(mag_bin_centers)[0]
    assert cosmo['h'] == temp_hod.hval, "BUG!"
    phi_L_cen_per_mag = phi_L_cen /delta_mag # h^3/Mpc^3/mag
    phi_L_sat_per_mag = phi_L_sat /delta_mag # h^3/Mpc^3/mag
    plot_phi_L_vs_M(mag_bin_centers, phi_L_cen_per_mag, phi_L_sat_per_mag, fsat, outname=f'{outdir}/Cacciato_HOD_phi_L_cen_sat_vs_M.pdf')

    # ---------------------------
    # binned Phi
    # ---------------------------
    # my bin choices at first
    # delta_mr = 0.1578 # this is the magnitude bin size quoted in Cacciato and seems like a typo
    # but my code gives 0.15625
    #outdir = "plots/my_bins_def"
    #bin_edges= np.linspace(-23, -18, 33)
    #"""
    #array([-23.     , -22.84375, -22.6875 , -22.53125, -22.375  , -22.21875,
    #   -22.0625 , -21.90625, -21.75   , -21.59375, -21.4375 , -21.28125,
    #   -21.125  , -20.96875, -20.8125 , -20.65625, -20.5    , -20.34375,
    #   -20.1875 , -20.03125, -19.875  , -19.71875, -19.5625 , -19.40625,
    #   -19.25   , -19.09375, -18.9375 , -18.78125, -18.625  , -18.46875,
    #   -18.3125 , -18.15625, -18.     ])
    #"""
    #delta_mag = np.diff(bin_edges)[0]
    #print("magnitude difference used to get per mag unit:", delta_mag)
    #mag_bins = zip(bin_edges[:-1], bin_edges[1:])

    # Closest definition of bins to match the Cacciato+19 plots
    num_bins = 32
    delta_mag = 0.1578
    # Build the centers directly so the anchor point matches exactly at -23
    bin_centers = -23.0 + np.arange(num_bins) * delta_mag
    """array([-23.    , -22.8422, -22.6844, -22.5266, -22.3688, -22.211 ,
           -22.0532, -21.8954, -21.7376, -21.5798, -21.422 , -21.2642,
           -21.1064, -20.9486, -20.7908, -20.633 , -20.4752, -20.3174,
           -20.1596, -20.0018, -19.844 , -19.6862, -19.5284, -19.3706,
           -19.2128, -19.055 , -18.8972, -18.7394, -18.5816, -18.4238,
           -18.266 , -18.1082])
    """
    # Generate corresponding edges array for HMF integration routines
    # The edges expand by half a bin width on either side of the centers
    bin_edges = np.zeros(num_bins + 1)
    # make bins with L+-dL/2, where dL = 0.1578
    bin_edges[:-1] = bin_centers - (delta_mag / 2.0) 
    bin_edges[-1]  = bin_centers[-1] + (delta_mag / 2.0)
    """
    array([-23.0789, -22.9211, -22.7633, -22.6055, -22.4477, -22.2899,
       -22.1321, -21.9743, -21.8165, -21.6587, -21.5009, -21.3431,
       -21.1853, -21.0275, -20.8697, -20.7119, -20.5541, -20.3963,
       -20.2385, -20.0807, -19.9229, -19.7651, -19.6073, -19.4495,
       -19.2917, -19.1339, -18.9761, -18.8183, -18.6605, -18.5027,
       -18.3449, -18.1871, -18.0293])
    """
    mag_bins = zip(bin_edges[:-1], bin_edges[1:])
    
    log_lum_bin_centers = np.empty(bin_edges.size - 1)
    mag_bin_centers = np.empty(bin_edges.size - 1)
    phi_L_cen = np.empty_like(log_lum_bin_centers)
    phi_L_sat = np.empty_like(log_lum_bin_centers)
    
    for ii, (M2, M1) in enumerate(mag_bins):
        log_L2 = np.log10(magnitude_to_luminosity(M2)) # bright #log10( L2 / [Lsun/h^2] )
        log_L1 = np.log10(magnitude_to_luminosity(M1)) # faint #log10( L1 / [Lsun/h^2] )
        log_lum_bin_centers[ii] = 0.5 * (log_L1 + log_L2)
        mag_bin_centers[ii] = 0.5 * (M1 + M2)  # Mr-5logh # only needed for plot labels
    
        my_params = dict(log_L1=log_L1, log_L2=log_L2, clfpars=cacciato_med_pars)
        temp_hod = hod_constructor(**my_params)
        # Integrate over the mass function to get phi(L) for centrals and satellites
        phi_L_cen[ii] = hmcalc.integrate_over_massfunc(
            temp_hod._Nc,
            cosmo,
            1 / (1 + 0.1),  # for LF, redshift fixed to 0.1 in Cacciato+2013
        )
        phi_L_sat[ii] = hmcalc.integrate_over_massfunc(
            temp_hod._Ns,
            cosmo,
            1 / (1 + 0.1),  # for LF, redshift fixed to 0.1 in Cacciato+2013
        ) 

    fsat = phi_L_sat / (phi_L_sat + phi_L_cen)

    binned_cen_hod = phi_L_cen / cosmo['h']**3 / delta_mag # h^3/Mpc^3/mag
    binned_sat_hod = phi_L_sat / cosmo['h']**3 / delta_mag # h^3/Mpc^3/mag
    print(f"binned cen HOD\n{binned_cen_hod}", f"binned sat HOD\n{binned_sat_hod}", sep="\n")
    plot_binned_LF(mag_bin_centers, binned_cen_hod, binned_sat_hod, fsat, outname=f"{outdir}/binned_LF_cacciato2013.pdf", save_png=True)

